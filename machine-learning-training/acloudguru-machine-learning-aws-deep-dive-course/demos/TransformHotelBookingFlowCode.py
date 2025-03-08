import base64
import collections
import copy
import io
import os
import re
import socket
import logging
import json
import hashlib
import numpy as np
import pandas as pd
import tempfile
import zipfile
from ipaddress import ip_address
from collections import Counter, namedtuple
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from enum import Enum
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import functions as sf, types, Column, Window
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import udf, pandas_udf, to_timestamp
from pyspark.sql.session import SparkSession
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    FloatType,
    FractionalType,
    IntegralType,
    LongType,
    StringType,
    TimestampType,
    StructType,
    ArrayType,
)
from pyspark.sql.utils import AnalysisException
from statsmodels.tsa.seasonal import STL


#  You may want to configure the Spark Context with the right credentials provider.
spark = SparkSession.builder.master("local").getOrCreate()
mode = None

JOIN_COLUMN_LIMIT = 10
DATAFRAME_AUTO_COALESCING_SIZE_THRESHOLD = 5368709120
ESCAPE_CHAR_PATTERN = re.compile("[{}]+".format(re.escape(".`")))
VALID_JOIN_TYPE = frozenset(
    [
        "anti",
        "cross",
        "full",
        "full_outer",
        "fullouter",
        "inner",
        "left",
        "left_anti",
        "left_outer",
        "left_semi",
        "leftanti",
        "leftouter",
        "leftsemi",
        "outer",
        "right",
        "right_outer",
        "rightouter",
        "semi",
    ],
)
DATE_SCALE_OFFSET_DESCRIPTION_SET = frozenset(["Business day", "Week", "Month", "Annual Quarter", "Year"])
DEFAULT_NODE_OUTPUT_KEY = "default"
OUTPUT_NAMES_KEY = "output_names"
SUPPORTED_TYPES = {
    BooleanType: "Boolean",
    FloatType: "Float",
    LongType: "Long",
    DoubleType: "Double",
    StringType: "String",
    DateType: "Date",
    TimestampType: "Timestamp",
}
JDBC_DEFAULT_NUMPARTITIONS = 2
KEY_JDBC_URL = "jdbcURL"
KEY_REFRESH_TOKEN = "refresh_token"
DEFAULT_RANDOM_SEED = 838257247
PREPROCESS_TEMP_TABLE_NAME = "DataWrangerPushdownTempTable"


MAX_NAME_LENGTH = 100
CUSTOM_UDF_MODE_PANDAS = "Pandas"
CUSTOM_UDF_MODE_PYTHON = "Python"


def capture_stdout(func, *args, **kwargs):
    """Capture standard output to a string buffer"""
    stdout_string = io.StringIO()
    with redirect_stdout(stdout_string):
        func(*args, **kwargs)
    return stdout_string.getvalue()


def convert_or_coerce(pandas_df, spark):
    """Convert pandas df to pyspark df and coerces the mixed cols to string"""
    try:
        return spark.createDataFrame(pandas_df)
    except TypeError as e:
        match = re.search(r".*field (\w+).*Can not merge type.*", str(e))
        if match is None:
            raise e
        mixed_col_name = match.group(1)
        # Coercing the col to string
        if mixed_col_name in pandas_df.columns:
            pandas_df[mixed_col_name] = pandas_df[mixed_col_name].astype("str")
        else:
            try:
                # If the column name is numeric
                mixed_col_name = float(mixed_col_name)
                pandas_df[mixed_col_name] = pandas_df[mixed_col_name].astype("str")
            except ValueError:
                raise e

        return pandas_df


def dedupe_columns(cols):
    """Dedupe and rename the column names after applying join operators. Rules:
        * First, append "_0", "_1" to dedupe and mark as renamed.
        * If the original df already takes the name, we will append more "_dup" as suffix til it's unique.
    """
    # spark by default is not case sensitive, so we will convert the column names to lower case
    lowered_cols = [col.lower() for col in cols]
    col_to_count = Counter(lowered_cols)
    duplicate_col_to_count = {col: col_to_count[col] for col in col_to_count if col_to_count[col] != 1}
    for i in range(len(lowered_cols)):
        col = lowered_cols[i]
        if col in duplicate_col_to_count:
            idx = col_to_count[col] - duplicate_col_to_count[col]
            new_col_name = f"{col}_{str(idx)}"
            while new_col_name in col_to_count:
                new_col_name += "_dup"
            cols[i] = new_col_name.replace(col, cols[i], 1)
            duplicate_col_to_count[col] -= 1
    return cols


def default_spark(value):
    return {DEFAULT_NODE_OUTPUT_KEY: value}


def default_spark_with_stdout(df, stdout):
    return {
        DEFAULT_NODE_OUTPUT_KEY: df,
        "stdout": stdout,
    }


def default_spark_with_trained_parameters(value, trained_parameters):
    return {DEFAULT_NODE_OUTPUT_KEY: value, "trained_parameters": trained_parameters}


def default_spark_with_trained_parameters_and_state(df, trained_parameters, state):
    return {DEFAULT_NODE_OUTPUT_KEY: df, "trained_parameters": trained_parameters, "state": state}


def dispatch(key_name, args, kwargs, funcs):
    """
    Dispatches to another operator based on a key in the passed parameters.
    This also slices out any parameters using the parameter_name passed in,
    and will reassemble the trained_parameters correctly after invocation.

    Args:
        key_name: name of the key in kwargs used to identify the function to use.
        args: dataframe that will be passed as the first set of parameters to the function.
        kwargs: keyword arguments that key_name will be found in; also where args will be passed to parameters.
                These are also expected to include trained_parameters if there are any.
        funcs: dictionary mapping from value of key_name to (function, parameter_name)
    """
    if key_name not in kwargs:
        raise OperatorCustomerError(f"Missing required parameter {key_name}")

    operator = kwargs[key_name]
    multi_column_operators = kwargs.get("multi_column_operators", [])

    if operator not in funcs:
        raise OperatorCustomerError(f"Invalid choice selected for {key_name}. {operator} is not supported.")

    func, parameter_name = funcs[operator]

    # Extract out the parameters that should be available.
    func_params = kwargs.get(parameter_name, {})
    if func_params is None:
        func_params = {}

    # Extract out any trained parameters.
    specific_trained_parameters = None
    if "trained_parameters" in kwargs:
        trained_parameters = kwargs["trained_parameters"]
        if trained_parameters is not None and parameter_name in trained_parameters:
            specific_trained_parameters = trained_parameters[parameter_name]
    func_params["trained_parameters"] = specific_trained_parameters

    result = spark_operator_with_escaped_column(
        func, args, func_params, multi_column_operators=multi_column_operators, operator_name=operator
    )

    # Check if the result contains any trained parameters and remap them to the proper structure.
    if result is not None and "trained_parameters" in result:
        existing_trained_parameters = kwargs.get("trained_parameters")
        updated_trained_parameters = result["trained_parameters"]

        if existing_trained_parameters is not None or updated_trained_parameters is not None:
            existing_trained_parameters = existing_trained_parameters if existing_trained_parameters is not None else {}
            existing_trained_parameters[parameter_name] = result["trained_parameters"]

            # Update the result trained_parameters so they are part of the original structure.
            result["trained_parameters"] = existing_trained_parameters
        else:
            # If the given trained parameters were None and the returned trained parameters were None, don't return
            # anything.
            del result["trained_parameters"]

    return result


def filter_timestamps_by_dates(df, timestamp_column, start_date=None, end_date=None):
    """Helper to filter dataframe by start and end date."""
    # ensure start date < end date, if both specified
    if start_date is not None and end_date is not None and pd.to_datetime(start_date) > pd.to_datetime(end_date):
        raise OperatorCustomerError(
            "Invalid combination of start and end date given. Start date should come before end date."
        )

    # filter by start date
    if start_date is not None:
        if pd.to_datetime(start_date) is pd.NaT:  # make sure start and end dates are datetime-castable
            raise OperatorCustomerError(
                f"Invalid start date given. Start date should be datetime-castable. Found: start date = {start_date}"
            )
        else:
            df = df.filter(
                sf.col(timestamp_column) >= sf.unix_timestamp(sf.lit(str(pd.to_datetime(start_date)))).cast("timestamp")
            )

    # filter by end date
    if end_date is not None:
        if pd.to_datetime(end_date) is pd.NaT:  # make sure start and end dates are datetime-castable
            raise OperatorCustomerError(
                f"Invalid end date given. Start date should be datetime-castable. Found: end date = {end_date}"
            )
        else:
            df = df.filter(
                sf.col(timestamp_column) <= sf.unix_timestamp(sf.lit(str(pd.to_datetime(end_date)))).cast("timestamp")
            )  # filter by start and end date

    return df


def format_sql_query_string(query_string):
    # Initial strip
    query_string = query_string.strip()

    # Remove semicolon.
    # This is for the case where this query will be wrapped by another query.
    query_string = query_string.rstrip(";")

    # Split lines and strip
    lines = query_string.splitlines()
    arr = []
    for line in lines:
        if not line.strip():
            continue
        line = line.strip()
        line = line.rstrip(";")
        arr.append(line)
    formatted_query_string = " ".join(arr)
    return formatted_query_string


def get_and_validate_join_keys(join_keys):
    join_keys_left = []
    join_keys_right = []
    for join_key in join_keys:
        left_key = join_key.get("left", "")
        right_key = join_key.get("right", "")
        if not left_key or not right_key:
            raise OperatorCustomerError("Missing join key: left('{}'), right('{}')".format(left_key, right_key))
        join_keys_left.append(left_key)
        join_keys_right.append(right_key)

    if len(join_keys_left) > JOIN_COLUMN_LIMIT:
        raise OperatorCustomerError("We only support join on maximum 10 columns for one operation.")
    return join_keys_left, join_keys_right


def get_dataframe_with_sequence_ids(df: DataFrame):
    df_cols = df.columns
    rdd_with_seq = df.rdd.zipWithIndex()
    df_with_seq = rdd_with_seq.toDF()
    df_with_seq = df_with_seq.withColumnRenamed("_2", "_seq_id_")
    for col_name in df_cols:
        df_with_seq = df_with_seq.withColumn(col_name, df_with_seq["_1"].getItem(col_name))
    df_with_seq = df_with_seq.drop("_1")
    return df_with_seq


def get_execution_state(status: str, message=None):
    return {"status": status, "message": message}


def get_trained_params_by_col(trained_params, col):
    if isinstance(trained_params, list):
        for params in trained_params:
            if params.get("input_column") == col:
                return params
        return None
    return trained_params


def get_unique_output_column_name(df_columns, output_column):
    idx = 0
    while output_column in df_columns:
        output_column = f"{output_column}_{idx}"
        idx += 1
    return output_column


def multi_output_spark(outputs_dict, handle_default=True):
    if handle_default and DEFAULT_NODE_OUTPUT_KEY in outputs_dict.keys():
        # Ensure 'default' is first in the list of output names if it is used
        output_names = [DEFAULT_NODE_OUTPUT_KEY]
        output_names.extend([key for key in outputs_dict.keys() if key != DEFAULT_NODE_OUTPUT_KEY])
    else:
        output_names = [key for key in outputs_dict.keys()]
    outputs_dict[OUTPUT_NAMES_KEY] = output_names
    return outputs_dict


def operator_in_list(operator: str, transform_list: List[str]):
    """Checks if operator is in the list of transforms"""
    return any([operator.startswith(transform) for transform in transform_list])


def rename_invalid_column(df, orig_col):
    """Rename a given column in a data frame to a new valid name

    Args:
        df: Spark dataframe
        orig_col: input column name

    Returns:
        a tuple of new dataframe with renamed column and new column name
    """
    temp_col = orig_col
    if ESCAPE_CHAR_PATTERN.search(orig_col):
        idx = 0
        temp_col = ESCAPE_CHAR_PATTERN.sub("_", orig_col)
        name_set = set(df.columns)
        while temp_col in name_set:
            temp_col = f"{temp_col}_{idx}"
            idx += 1
        df = df.withColumnRenamed(orig_col, temp_col)
    return df, temp_col


def spark_operator_with_escaped_column(
    operator_func,
    func_args,
    func_params,
    multi_column_operators=[],
    operator_name="",
    output_name=DEFAULT_NODE_OUTPUT_KEY,
):
    """Invoke operator func with input dataframe that has its column names sanitized.

    This function renames column names with special char to an internal name and
    rename it back after invocation

    Args:
        operator_func: underlying operator function
        func_args: operator function positional args, this only contains one element `df` for now
        func_params: operator function kwargs
        multi_column_operators: list of operators that support multiple columns with iteration, value of '*' indicates
        support all. Note: it doesn't include operators that support single column and multiple columns in parallel (e.g., imputing numeric missing).
        operator_name: operator name defined in node parameters
        output_name: the name of the output in the operator function result

    Returns:
        a dictionary with operator results
    """
    renamed_columns = {}
    iterate_over_multiple_columns = False
    input_column_key = "input_column"
    valid_output_column_keys = {"output_column", "output_prefix", "output_column_prefix"}
    is_output_col_key = set(func_params.keys()).intersection(valid_output_column_keys)
    output_column_key = list(is_output_col_key)[0] if is_output_col_key else None
    output_trained_params = []

    # Copy on write so the original func_params is untouched to ensure inference mode correctness
    func_params = func_params.copy()
    for parameter_name, parameter_value in func_params.items():
        if parameter_name != "trained_parameters":
            func_params[parameter_name] = copy.deepcopy(parameter_value)

    if input_column_key in func_params:

        # Convert input_columns to list if string ensuring backwards compatibility with strings
        input_columns = (
            func_params[input_column_key]
            if isinstance(func_params[input_column_key], list)
            else [func_params[input_column_key]]
        )

        # rename columns if needed
        sanitized_input_columns = []
        for input_col_value in input_columns:
            input_df, temp_col_name = rename_invalid_column(func_args[0], input_col_value)
            func_args[0] = input_df
            if temp_col_name != input_col_value:
                renamed_columns[input_col_value] = temp_col_name
            sanitized_input_columns.append(temp_col_name)

        func_params[input_column_key] = (
            sanitized_input_columns if isinstance(func_params[input_column_key], list) else sanitized_input_columns[0]
        )

        trained_params_mul_cols = func_params.get("trained_parameters")
        TRAINED_PARAMS_BACKWARDS_COMPATABILITY_CONDITION = (
            trained_params_mul_cols
            and isinstance(trained_params_mul_cols, list)
            and any(key in trained_params_mul_cols[0] for key in TRAINED_PARAMS_KEYS_FOR_BACKWARD_COMPATABILITY)
        )
        iterate_over_multiple_columns = (
            any(op_name in multi_column_operators for op_name in ["*", operator_name])
            or TRAINED_PARAMS_BACKWARDS_COMPATABILITY_CONDITION
        )
        output_column_name = func_params.get(output_column_key)
        append_column_name_to_output_column = len(input_columns) > 1 and output_column_name
        result = None

        if iterate_over_multiple_columns:
            # invalidate trained params if not type list for multi-column use case
            if len(sanitized_input_columns) > 1 and isinstance(trained_params_mul_cols, dict):
                trained_params_mul_cols = func_params["trained_parameters"] = None
            # output_column name as prefix if
            # 1. there are multiple input columns
            # 2. the output_column_key exists in params
            # 3. the output_column_value is not an empty string

            for input_col_val in sanitized_input_columns:
                if trained_params_mul_cols:
                    func_params["trained_parameters"] = get_trained_params_by_col(
                        trained_params_mul_cols, input_col_val
                    )
                func_params[input_column_key] = input_col_val
                # if more than 1 column, output column name behaves as a prefix,
                if append_column_name_to_output_column:
                    func_params[output_column_key] = f"{output_column_name}_{input_col_val}"

                # invoke underlying function on each column if multiple are present
                result = operator_func(*func_args, **func_params)
                func_args[0] = result[output_name]

                if result.get("trained_parameters"):
                    # add input column to remove dependency on list order
                    trained_params_copy = result["trained_parameters"].copy()
                    trained_params_copy["input_column"] = input_col_val
                    output_trained_params.append(trained_params_copy)

        else:
            # if more than 1 column, output column name behaves as a prefix,
            if append_column_name_to_output_column:
                func_params[output_column_key] = (
                    [f"{output_column_name}_{input_col_val}" for input_col_val in func_params[input_column_key]]
                    if isinstance(func_params[input_column_key], list)
                    else f"{output_column_name}_{func_params[input_column_key]}"
                )

            result = operator_func(*func_args, **func_params)
            func_args[0] = result[output_name]
    else:
        # invoke underlying function
        result = operator_func(*func_args, **func_params)

    # put renamed columns back if applicable
    if result is not None and output_name in result:
        result_df = result[output_name]
        # rename col
        for orig_col_name, temp_col_name in renamed_columns.items():
            if temp_col_name in result_df.columns:
                result_df = result_df.withColumnRenamed(temp_col_name, orig_col_name)
        result[output_name] = result_df

    if len(output_trained_params) > 1:
        result["trained_parameters"] = output_trained_params

    return result


def stl_decomposition(ts, period=None):
    """Completes a Season-Trend Decomposition using LOESS (Cleveland et. al. 1990) on time series data.

    Parameters
    ----------
    ts: pandas.Series, index must be datetime64[ns] and values must be int or float.
    period: int, primary periodicity of the series. Default is None, will apply a default behavior
        Default behavior:
            if timestamp frequency is minute: period = 1440 / # of minutes between consecutive timestamps
            if timestamp frequency is second: period = 3600 / # of seconds between consecutive timestamps
            if timestamp frequency is ms, us, or ns: period = 1000 / # of ms/us/ns between consecutive timestamps
            else: defer to statsmodels' behavior, detailed here:
                https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/tsatools.py#L776

    Returns
    -------
    season: pandas.Series, index is same as ts, values are seasonality of ts
    trend: pandas.Series, index is same as ts, values are trend of ts
    resid: pandas.Series, index is same as ts, values are the remainder (original signal, subtract season and trend)
    """
    # TODO: replace this with another, more complex method for finding a better period
    period_sub_hour = {
        "T": 1440,  # minutes
        "S": 3600,  # seconds
        "M": 1000,  # milliseconds
        "U": 1000,  # microseconds
        "N": 1000,  # nanoseconds
    }
    if period is None:
        freq = ts.index.freq
        if freq is None:
            freq = pd.tseries.frequencies.to_offset(pd.infer_freq(ts.index))
        if freq is None:  # if still none, datetimes are not uniform, so raise error
            raise OperatorCustomerError(
                f"No uniform datetime frequency detected. Make sure the column contains datetimes that are evenly spaced (Are there any missing values?)"
            )
        for k, v in period_sub_hour.items():
            # if freq is not in period_sub_hour, then it is hourly or above and we don't have to set a default
            if k in freq.name:
                period = int(v / int(freq.n))  # n is always >= 1
                break
    model = STL(ts, period=period)
    decomposition = model.fit()
    return decomposition.seasonal, decomposition.trend, decomposition.resid, model.period


def to_vector(df, array_column):
    """Helper function to convert the array column in df to vector type column"""
    _udf = sf.udf(lambda r: Vectors.dense(r), VectorUDT())
    df = df.withColumn(array_column, _udf(array_column))
    return df


def uniform_sample(df, target_example_num, n_rows=None, min_required_rows=None):
    if n_rows is None:
        n_rows = df.count()
    if min_required_rows and n_rows < min_required_rows:
        raise OperatorCustomerError(
            f"Not enough valid rows available. Expected a minimum of {min_required_rows}, but the dataset contains "
            f"only {n_rows}"
        )
    sample_ratio = min(1, 3.0 * target_example_num / n_rows)
    return df.sample(withReplacement=False, fraction=float(sample_ratio), seed=0).limit(target_example_num)


def use_scientific_notation(values):
    """
    Return whether or not to use scientific notation in visualization's y-axis.

    Parameters
    ----------
    values: numpy array of values being plotted

    Returns
    -------
    boolean, True if viz should use scientific notation, False if not
    """
    _min = np.min(values)
    _max = np.max(values)
    _range = abs(_max - _min)
    return not (
        _range > 1e-3 and _range < 1e3 and abs(_min) > 1e-3 and abs(_min) < 1e3 and abs(_max) > 1e-3 and abs(_max) < 1e3
    )


def validate_col_name_in_df(col, df_cols):
    if col not in df_cols:
        raise OperatorCustomerError("Cannot resolve column name '{}'.".format(col))


def validate_join_type(join_type):
    if join_type not in VALID_JOIN_TYPE:
        raise OperatorCustomerError(
            "Unsupported join type '{}'. Supported join types include: {}.".format(
                join_type, ", ".join(VALID_JOIN_TYPE)
            )
        )


class OperatorCustomerError(Exception):
    """Error type for Customer Errors in Spark Operators"""


import boto3
import json
import logging
import os

from botocore.exceptions import ClientError
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException



def write_df_to_s3(df, output_config: dict):

    path = output_config["output_path"].replace("s3://", "s3a://")
    output_content_type = output_config["output_content_type"]


    partition_config = output_config["partition_config"] if output_config.get("partition_config") else {}

    num_partitions = partition_config.get("num_partitions")
    try:
        num_partitions = int(num_partitions) if num_partitions else None
    except ValueError as e:
        raise OperatorCustomerError(
            "The value of `num_partitions` in the partition configuration must be an "
            f"integer. Provide a value between 1 and 9999, and try running a job again.\n{e}"
        )
    if num_partitions and (num_partitions < 1 or num_partitions > 9999):
        limited = max(min(9999, num_partitions), 1)
        logging.warning(f"num_partitions {num_partitions} exceeds bounds: [1, 9999]. Limiting the value to {limited}")
        num_partitions = limited

    partition_by = partition_config.get("partition_by")
    partition_by = [partition_by] if isinstance(partition_by, str) else partition_by
    df_schema = [field.name for field in df.schema.fields]
    if partition_by:
        try:
            if not isinstance(partition_by, list):
                raise ValueError(partition_by)
            if not all([column in df_schema for column in partition_by]):
                raise OperatorCustomerError(
                    "The column names that you provided in the partition configuration, `partition_by`, did "
                    "not match the schema of the transformed dataset. Fix the partition configuration and "
                    f"try running a job again.\nColumn names: {partition_by}\nDataset schema: {df_schema}"
                )
        except Exception as e:
            raise OperatorCustomerError(
                "The value of `partition_by` in the partition configuration must be a list of strings. "
                f"Provide your column names as a list of strings and try running a job again.\n{e}"
            )

    updated_df = coalesce_df(df, partitions=num_partitions)
    compression = output_config["compression"]

    if output_content_type == "CSV":
        delimiter = output_config.get("delimiter", ",")
        write_options = (
            serialize_columns(updated_df)
            .write.option("nullValue", None)
            .option("compression", compression)
            .option("delimiter", delimiter)
            .option("header", "true")
            .option("escape", '"')
            .option("quote", '"')
            .format("csv")
        )
    elif output_content_type == "PARQUET":
        write_options = updated_df.write.option("compression", compression).format("parquet")
    else:
        raise OperatorCustomerError(
            f"'{output_content_type}' is not a valid content type. Use one of the following "
            "output file formats and try your request again: 'CSV', 'Parquet'"
        )

    if partition_by:
        write_options = write_options.partitionBy(*partition_by)

    try:
        write_options.save(path)
    except AnalysisException as e:
        raise OperatorCustomerError(e)

    logging.info(f"S3 output path: {path}")
    return default_spark(updated_df)


def write_visualizations_to_s3(visualizations: dict, output_config: dict, spark: SparkSession):
    """
    Write visualizations to s3

    visualizations: dict, output of visualizations
    output_config: dict, configurations for PJ
    """
    # Default to Json format
    serialization_format = output_config.get("output_content_type", "JSON")
    path = output_config["output_path"]

    bucket_name, prefix = s3_parse_bucket_name_and_prefix(path)

    region_name = os.getenv("AWS_REGION")
    if region_name:
        s3_client = boto3.client("s3", region_name=region_name)
    else:
        s3_client = boto3.client("s3")

    sse_algorithm = spark.sparkContext._jsc.hadoopConfiguration().get("fs.s3a.server-side-encryption-algorithm")
    output_kms_key = spark.sparkContext._jsc.hadoopConfiguration().get("fs.s3a.server-side-encryption.key")

    # Write to json and upload
    viz_json = json.dumps(visualizations, default=str)
    # TODO: sim https://sim.amazon.com/issues/SDW-5321
    file_name = "data_wrangler_visualization_job.json"
    put_object_args = {
        "Bucket": bucket_name,
        "Key": os.path.join(prefix, file_name),
        "Body": viz_json,
    }
    if sse_algorithm == "SSE-KMS" and output_kms_key:
        put_object_args["ServerSideEncryption"] = "aws:kms"
        put_object_args["SSEKMSKeyId"] = output_kms_key

    try:
        s3_client.put_object(**put_object_args)
    except ClientError as err:
        # https://docs.aws.amazon.com/AmazonS3/latest/API/ErrorResponses.html
        # 400 - bad request, 403 - forbidden, 404 - Not Found
        response_code = err.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if response_code:
            if 400 <= response_code < 500:
                raise OperatorCustomerError(
                    f"The {file_name} file couldn't be uploaded to S3 successfully. Use the following messages to fix your "
                    f"error and try your request again: \n\n{err.response.get('Error').get('Message')}"
                )

        # Raise errors for 5xx
        raise RuntimeError(f"An error occurred when writing visualization output to S3: {err}")
    except Exception as err:
        raise RuntimeError(f"An error occurred when writing visualization output to S3: {err}")

    return default_spark(visualizations)


from pyspark.ml.feature import NGram, HashingTF, MinHashLSH, MinHashLSHModel
from pyspark.sql import functions as sf
from pyspark.sql import types
from pyspark.sql.types import StringType
from pyspark.ml.functions import vector_to_array

import numpy as np


OUTPUT_STYLE_VECTOR = "Vector"
OUTPUT_STYLE_COLUMNS = "Columns"


def encode_categorical_ordinal_encode(
    df, input_column=None, output_column=None, invalid_handling_strategy=None, trained_parameters=None
):
    INVALID_HANDLING_STRATEGY_SKIP = "Skip"
    INVALID_HANDLING_STRATEGY_ERROR = "Error"
    INVALID_HANDLING_STRATEGY_KEEP = "Keep"
    INVALID_HANDLING_STRATEGY_REPLACE_WITH_NAN = "Replace with NaN"

    from pyspark.ml.feature import StringIndexer, StringIndexerModel
    from pyspark.sql.functions import when

    expects_column(df, input_column, "Input column")

    invalid_handling_map = {
        INVALID_HANDLING_STRATEGY_SKIP: "skip",
        INVALID_HANDLING_STRATEGY_ERROR: "error",
        INVALID_HANDLING_STRATEGY_KEEP: "keep",
        INVALID_HANDLING_STRATEGY_REPLACE_WITH_NAN: "keep",
    }

    output_column, output_is_temp = get_temp_col_if_not_set(df, output_column)

    # process inputs
    handle_invalid = (
        invalid_handling_strategy
        if invalid_handling_strategy in invalid_handling_map
        else INVALID_HANDLING_STRATEGY_ERROR
    )

    trained_parameters = load_trained_parameters(
        trained_parameters, {"invalid_handling_strategy": invalid_handling_strategy}
    )

    input_handle_invalid = invalid_handling_map.get(handle_invalid)
    index_model, index_model_loaded = load_pyspark_model_from_trained_parameters(
        trained_parameters, StringIndexerModel, "string_indexer_model"
    )

    if index_model is None:
        indexer = StringIndexer(inputCol=input_column, outputCol=output_column, handleInvalid=input_handle_invalid)
        # fit the model and transform
        try:
            index_model = fit_and_save_model(trained_parameters, "string_indexer_model", indexer, df)
        except Exception as e:
            if input_handle_invalid == "error":
                raise OperatorCustomerError(
                    f"Encountered error calculating string indexes. Halting because error handling is set to 'Error'. Please check your data and try again: {e}"
                )
            else:
                raise e
    trained_parameters["encoder_mapping"] = dict(enumerate(index_model.labelsArray[0]))

    output_df = transform_using_trained_model(index_model, df, index_model_loaded)

    # finally, if missing should be nan, convert them
    if handle_invalid == INVALID_HANDLING_STRATEGY_REPLACE_WITH_NAN:
        new_val = float("nan")
        # convert all numLabels indices to new_val
        num_labels = len(index_model.labelsArray[0])
        output_df = output_df.withColumn(
            output_column, when(output_df[output_column] == num_labels, new_val).otherwise(output_df[output_column])
        )

    # finally handle the output column name appropriately.
    output_df = replace_input_if_output_is_temp(output_df, input_column, output_column, output_is_temp)

    return default_spark_with_trained_parameters(output_df, trained_parameters)


def encode_categorical_one_hot_encode(
    df,
    input_column=None,
    input_already_ordinal_encoded=None,
    invalid_handling_strategy=None,
    drop_last=None,
    output_style=None,
    output_column=None,
    trained_parameters=None,
):
    INVALID_HANDLING_STRATEGY_SKIP = "Skip"
    INVALID_HANDLING_STRATEGY_ERROR = "Error"
    INVALID_HANDLING_STRATEGY_KEEP = "Keep"

    invalid_handling_map = {
        INVALID_HANDLING_STRATEGY_SKIP: "skip",
        INVALID_HANDLING_STRATEGY_ERROR: "error",
        INVALID_HANDLING_STRATEGY_KEEP: "keep",
    }

    handle_invalid = invalid_handling_map.get(invalid_handling_strategy, "error")
    expects_column(df, input_column, "Input column")
    output_format = output_style if output_style in [OUTPUT_STYLE_VECTOR, OUTPUT_STYLE_COLUMNS] else OUTPUT_STYLE_VECTOR
    drop_last = parse_parameter(bool, drop_last, "Drop Last", True)
    input_ordinal_encoded = parse_parameter(bool, input_already_ordinal_encoded, "Input already ordinal encoded", False)

    output_column = output_column if output_column else input_column

    trained_parameters = load_trained_parameters(
        trained_parameters, {"invalid_handling_strategy": invalid_handling_strategy, "drop_last": drop_last}
    )

    from pyspark.ml.feature import (
        StringIndexer,
        StringIndexerModel,
        OneHotEncoder,
        OneHotEncoderModel,
    )

    # first step, ordinal encoding. Not required if input_ordinal_encoded==True
    # get temp name for ordinal encoding
    ordinal_name = temp_col_name(df, output_column)
    if input_ordinal_encoded:
        df_ordinal = df.withColumn(ordinal_name, df[input_column].cast("int"))
        labels = None
    else:
        index_model, index_model_loaded = load_pyspark_model_from_trained_parameters(
            trained_parameters, StringIndexerModel, "string_indexer_model"
        )
        if index_model is None:
            # one hot encoding in PySpark will not work with empty string, replace it with null values
            df = df.withColumn(input_column, sf.when(sf.col(input_column) == "", None).otherwise(sf.col(input_column)))
            # apply ordinal encoding
            indexer = StringIndexer(inputCol=input_column, outputCol=ordinal_name, handleInvalid=handle_invalid)
            try:
                index_model = fit_and_save_model(trained_parameters, "string_indexer_model", indexer, df)
            except Exception as e:
                if handle_invalid == "error":
                    raise OperatorCustomerError(
                        f"Encountered error calculating string indexes. Halting because error handling is set to 'Error'. Please check your data and try again: {e}"
                    )
                else:
                    raise e

        try:
            df_ordinal = transform_using_trained_model(index_model, df, index_model_loaded)
        except Exception as e:
            if handle_invalid == "error":
                raise OperatorCustomerError(
                    f"Encountered error transforming string indexes. Halting because error handling is set to 'Error'. Please check your data and try again: {e}"
                )
            else:
                raise e

        labels = index_model.labels

    # drop the input column if required from the ordinal encoded dataset
    if output_column == input_column:
        df_ordinal = df_ordinal.drop(input_column)

    temp_output_col = temp_col_name(df_ordinal, output_column)

    # apply onehot encoding on the ordinal
    cur_handle_invalid = handle_invalid if input_ordinal_encoded else "error"
    cur_handle_invalid = "keep" if cur_handle_invalid == "skip" else cur_handle_invalid

    ohe_model, ohe_model_loaded = load_pyspark_model_from_trained_parameters(
        trained_parameters, OneHotEncoderModel, "one_hot_encoder_model"
    )
    if ohe_model is None:
        ohe = OneHotEncoder(
            dropLast=drop_last, handleInvalid=cur_handle_invalid, inputCol=ordinal_name, outputCol=temp_output_col
        )
        try:
            ohe_model = fit_and_save_model(trained_parameters, "one_hot_encoder_model", ohe, df_ordinal)
        except Exception as e:
            if handle_invalid == "error":
                raise OperatorCustomerError(
                    f"Encountered error calculating encoding categories. Halting because error handling is set to 'Error'. Please check your data and try again: {e}"
                )
            else:
                raise e

    output_df = transform_using_trained_model(ohe_model, df_ordinal, ohe_model_loaded)

    if output_format == OUTPUT_STYLE_COLUMNS:
        if labels is None:
            labels = list(range(ohe_model.categorySizes[0]))

        current_output_cols = set(list(output_df.columns))
        old_cols = [sf.col(escape_column_name(name)) for name in df.columns if name in current_output_cols]
        arr_col = vector_to_array(output_df[temp_output_col])
        new_cols = [(arr_col[i]).alias(f"{output_column}_{name}") for i, name in enumerate(labels)]
        output_df = output_df.select(*(old_cols + new_cols))
    else:
        # remove the temporary ordinal encoding
        output_df = output_df.drop(ordinal_name)
        output_df = output_df.withColumn(output_column, sf.col(temp_output_col))
        output_df = output_df.drop(temp_output_col)
        final_ordering = [col for col in df.columns]
        if output_column not in final_ordering:
            final_ordering.append(output_column)

        final_ordering = escape_column_names(final_ordering)
        output_df = output_df.select(final_ordering)

    return default_spark_with_trained_parameters(output_df, trained_parameters)


def encode_categorical_similarity_encode(
    df, input_column, output_column=None, target_dimension=30, output_style=None, trained_parameters=None
):
    """
    Encode a categorical variable with similarity encoding.
    This technique works when the number of categories is large, or if the data is noisy. The encoding takes the
    category names into account and assigns similar embedding vectors to categories with similar names (e.g.
    "table (brown)" and "table (gray)" or "California" and "Califronia").
    It is based on a paper "Encoding high-cardinality string categorical variables, P. Cedra and G. Varoquaux". A
    category is converted to a collection of tokens obtained from 3-gram on the character level. Each such token set is
    converted into a numeric vector via the min-hash encoding. This encoding makes sure that collections with a large
    intersection result in vectors with a large number of equal elements.

    Args:
        df: Input dataframe
        input_column: Column containing the categorical variable
        output_column: Depending on the output style, this is either the name of the output column or the prefix for the
            output columns.
        target_dimension: The dimension of the embedding vector of the encoding.
        output_style: Either "Vector" or "Columns". If "Vector" the output is a single column where each entry is a list
            of numbers. If "Columns", we create a new column for every dimension of the embedding.
        trained_parameters: If the transform was previously fit, this contain the encoding of the created spark models.

    Returns:
        returns both the trained_parameters containing an encoding of the spark models created, and the dataframe with
        the new column or columns containing the category embedding

    """
    # set up parameters
    expects_column(df, input_column, "Input column")
    output_format = output_style if output_style in [OUTPUT_STYLE_VECTOR, OUTPUT_STYLE_COLUMNS] else OUTPUT_STYLE_VECTOR
    if output_column:
        tmp_output = output_column
    else:
        output_column = input_column
        tmp_output = temp_col_name(df)

    # Check if input_col is has the supported (string) type.
    column_type = df.schema[input_column].dataType
    if not isinstance(column_type, StringType):
        raise OperatorCustomerError(
            f"Unsupported data type for input column: {column_type}. "
            "We currently support only inputs of String type. Select a column or convert the column you've selected to "
            f"{StringType}."
        )

    # load trained parameters
    trained_parameters = load_trained_parameters(trained_parameters, {"target_dimension": target_dimension})

    # convert the categorical column into tokenized bag of 3-gram characters
    ngram_col = temp_col_name(df, tmp_output)
    df_ngram = _tokenize_char_ngram(df, input_column, ngram_col)
    # encode the tokens via minhash, and drop the temporary ngram column
    df_minhash, trained_parameters = _min_hash_tokens(
        df_ngram, ngram_col, tmp_output, target_dimension, trained_parameters
    )
    df_minhash = df_minhash.drop(ngram_col)

    # finalize the output
    if output_format == OUTPUT_STYLE_COLUMNS:
        labels = list(range(target_dimension))
        current_output_cols = set(list(df_minhash.columns)) - {tmp_output}
        old_cols = [sf.col(escape_column_name(name)) for name in df_minhash.columns if name in current_output_cols]
        new_cols = [(sf.col(tmp_output)[i]).alias(f"{output_column}_{i}") for i in labels]
        df_out = df_minhash.select(*(old_cols + new_cols))
    else:
        df_out = df_minhash
        if tmp_output != output_column:
            df_out = df_out.drop(input_column).withColumnRenamed(tmp_output, output_column)

    return default_spark_with_trained_parameters(df_out, trained_parameters)


def _tokenize_char_ngram(df, text_col, ngarm_col, ngram_size=3):
    """
    Tokenizes text column into character ngrams.
    Before tokenizing into ngrams, the following preprocessing is done:
    1. Missing and empty strings are converted to a string containing a single space.
    2. n - 2 spaces are added to the beginning of the string
    3. Letters are lowercased
    4. Any consecutive sequence of spaces, tabs, line breaks, is converted to a single space.

    Steps 1,2 ensure the output is never an empty list. Steps 3,4 provide basic data cleaning

    Args:
        df: input dataframe
        text_col: name of input column with text data
        ngarm_col: name of output column
        ngram_size: size of ngrams (default 3)

    Returns:
        dataframe with a new column. This column contains arrays with the ngrams corresponding to the input text

    """
    df = df.fillna(" ", subset=[text_col])

    text_lower = sf.lower(df[text_col])
    text_spacing = sf.regexp_replace(text_lower, "\\s+", " ")
    no_empty = sf.when(text_spacing == "", " ").otherwise(text_spacing)
    # The following helps (1) avoid outputting empty lists for short strings (2) add ngrams that capture the first
    # tokens, that tend to be important
    if ngram_size >= 3:
        padded = sf.concat(sf.lit(" " * (ngram_size - 2)), no_empty)
    else:
        padded = no_empty
    char_list = sf.split(padded, pattern="")

    char_list_col = temp_col_name(df, ngarm_col)
    df = df.withColumn(char_list_col, char_list)

    ngram = NGram(n=ngram_size, inputCol=char_list_col, outputCol=ngarm_col)
    df = ngram.transform(df)
    df = df.drop(char_list_col)
    return df


def _min_hash_tokens(df, token_col, out_col, target_dimension, trained_parameters):
    # first convert token arrays to a high dimensional sparse vector via hashing. Specifically, every token is mapped
    # via a hash function to an index. The value of the vector at that index will be 1. The value will remain 1 even
    # if the same token appears more than once.
    sparse_vec_col = temp_col_name(df)
    hasher = HashingTF(inputCol=token_col, outputCol=sparse_vec_col, binary=True)
    df = hasher.transform(df)

    # prepare minhash model. When the transform is not applied for the first time (i.e. we are in transform, not
    # fit_transform mode), the model should be loaded from the trained parameters. Otherwise, it will be None and
    # created.
    minhash_out = temp_col_name(df, out_col)
    minhash_model, minhash_model_loaded = load_pyspark_model_from_trained_parameters(
        trained_parameters, MinHashLSHModel, "minhash_model"
    )
    if minhash_model is None:
        # apply ordinal encoding
        mh = MinHashLSH(inputCol=sparse_vec_col, outputCol=minhash_out, numHashTables=target_dimension)
        minhash_model = fit_and_save_model(trained_parameters, "minhash_model", mh, df)

    # apply minhash model and get rid of the sparse represnetation
    df_with_hash = transform_using_trained_model(minhash_model, df, minhash_model_loaded).drop(sparse_vec_col)

    # the output of the minhash model is in an inconvenient format: An array of vectors of length 1. Convert each such
    # array to a numpy array. Also normalize the numbers to be in [-1,1]
    min_val = 0
    max_val = np.iinfo(np.int32).max
    divisor = (max_val - min_val) / 2
    subtract = min_val + divisor

    # TODO: Performance could potentially be increased by avoiding a udf. The challenge is the format of the input: A
    #  list of Vector objects, each having a single element.
    @sf.udf(returnType=types.ArrayType(types.DoubleType()))
    def to_np(densevec_list):
        return [(float(x[0]) - subtract) / divisor for x in densevec_list]

    df_out = df_with_hash.withColumn(out_col, to_np(minhash_out)).drop(minhash_out)
    return df_out, trained_parameters


import datetime

import pyspark.sql.functions as sf
from pyspark.ml.feature import Imputer, ImputerModel
from pyspark.sql.types import (
    DoubleType,
    StringType,
    IntegralType,
    NumericType,
    IntegerType,
    ShortType,
    LongType,
    ByteType,
    FloatType,
    DecimalType,
    DateType,
    TimestampType,
    BooleanType,
)



#  numeric types for handle missing operator
NUMERIC_DATATYPES = {IntegerType, ShortType, LongType, DoubleType, FloatType, ByteType, DecimalType}


def handle_missing_get_indicator_column(df, input_column, expected_type):
    """Helper function used to get an indicator for all missing values."""
    dcol = df[input_column].cast(expected_type)
    if isinstance(expected_type, StringType):
        indicator = sf.isnull(dcol) | (sf.trim(dcol) == "")
    elif isinstance(expected_type, (DateType, TimestampType, BooleanType)):
        indicator = sf.col(input_column).isNull()
    else:
        indicator = sf.isnull(dcol) | sf.isnan(dcol)
    return indicator


def handle_missing_replace_missing_values(df, input_column, output_column, impute_value, expected_type):
    """Helper function that replaces any missing values with the impute value."""

    expects_column(df, input_column, "Input column")

    if not isinstance(expected_type, (DoubleType, StringType, LongType, DateType, TimestampType)):
        raise OperatorCustomerError(f"Canvas does not support imputation for type {expected_type.typeName()}.")
    # Set output to default to input column if None or empty
    output_column = input_column if not output_column else output_column

    # Create a temp missing indicator column
    missing_col = temp_col_name(df)
    try:
        output_df = df.withColumn(missing_col, handle_missing_get_indicator_column(df, input_column, expected_type))
    except OverflowError as err:
        raise OperatorCustomerError(f"Value in column {input_column} is out of range.")

    # Fill values and drop the temp indicator column

    output_df = output_df.withColumn(
        output_column,
        sf.when(output_df[missing_col] == 0, output_df[input_column]).otherwise(impute_value).cast(expected_type),
    ).drop(missing_col)

    return output_df


def handle_missing_numeric(df, input_column=None, output_column=None, strategy=None, trained_parameters=None):
    impute_strategy = {
        "Mean": "mean",
        "Approximate Median": "median",
    }

    # Validate column name and type
    expects_column(df, input_column, "Input column")
    input_column_list = input_column if isinstance(input_column, list) else [input_column]
    output_column_list = (
        output_column
        if output_column is None or output_column == "" or isinstance(output_column, list)
        else [output_column]
    )
    for col in input_column_list:
        if not isinstance(df.schema[col].dataType, tuple(NUMERIC_DATATYPES)):
            raise OperatorCustomerError(
                f"Canvas can't calculate the imputation value for the column {col}. "
                f"Choose a numeric column or select 'Categorical' for 'Column type'."
            )

    trained_parameters = load_trained_parameters(trained_parameters, {"strategy": strategy})

    # Support existing data flows for backward compatibility
    if trained_parameters and "impute_value" in trained_parameters:
        impute_value = parse_parameter(
            float, trained_parameters.get("impute_value"), "Trained parameters", nullable=True
        )
        if impute_value:
            output_df = handle_missing_replace_missing_values(
                df, input_column, output_column, impute_value, DoubleType()
            )
            return default_spark_with_trained_parameters(output_df, trained_parameters)

    # Impute missing with ImputerModel
    imputer_model, imputer_model_loaded, temp_dir_path = load_pyspark_model_from_trained_parameters_and_store_artifacts(
        trained_parameters, ImputerModel, "imputer_model"
    )
    if imputer_model is None:
        if strategy in impute_strategy:
            imputer = Imputer(
                inputCols=input_column_list,
                outputCols=output_column_list if output_column_list else input_column_list,
                strategy=impute_strategy[strategy],
            )
            imputer_model = fit_and_save_model(trained_parameters, "imputer_model", imputer, df)
        else:
            raise OperatorInternalError(
                f"Unexpected things happened. Invalid imputation strategy specified: {strategy}"
            )

    output_df = transform_using_trained_model_and_clean_artifacts(
        imputer_model, df, imputer_model_loaded, temp_dir_path
    )

    return default_spark_with_trained_parameters(output_df, trained_parameters)


def handle_missing_categorical(df, input_column=None, output_column=None, trained_parameters=None):
    # validate column  name and type
    expects_column(df, input_column, "Input column")
    expected_type = df.schema[input_column].dataType
    single_col = df.select(input_column).filter(~handle_missing_get_indicator_column(df, input_column, expected_type))
    trained_parameters = load_trained_parameters(trained_parameters, {})
    impute_value = parse_parameter(str, trained_parameters.get("impute_value"), "Trained parameters", nullable=True)
    date_type_mapping = {DateType: datetime.date, TimestampType: datetime.datetime}

    if impute_value and isinstance(expected_type, (TimestampType, DateType)):
        impute_value = date_type_mapping[expected_type.__class__].fromisoformat(impute_value)
    elif impute_value is None:
        try:
            top2counts = single_col.groupby(input_column).count().sort("count", ascending=False).head(2)
            impute_value = None
            for row in top2counts:
                if row[input_column] is not None:
                    impute_value = row[input_column]
                    break
            if isinstance(expected_type, (TimestampType, DateType)):
                trained_parameters["impute_value"] = date_type_mapping[expected_type.__class__].isoformat(impute_value)
            else:
                trained_parameters["impute_value"] = impute_value
        except Exception:
            raise OperatorCustomerError(
                f"Could not calculate imputation value. Please ensure your column contains multiple values."
            )

    output_df = handle_missing_replace_missing_values(df, input_column, output_column, impute_value, expected_type)

    return default_spark_with_trained_parameters(output_df, trained_parameters)


def handle_missing_impute(df, **kwargs):
    kwargs["multi_column_operators"] = ["Categorical"]
    return dispatch(
        "column_type",
        [df],
        kwargs,
        {
            "Numeric": (handle_missing_numeric, "numeric_parameters"),
            "Categorical": (handle_missing_categorical, "categorical_parameters"),
        },
    )


def handle_missing_fill_missing(df, input_column=None, output_column=None, fill_value=None, trained_parameters=None):
    expects_column(df, input_column, "Input column")
    if isinstance(df.schema[input_column].dataType, IntegralType):
        fill_value = parse_parameter(int, fill_value, "Fill Value")
    elif isinstance(df.schema[input_column].dataType, NumericType):
        fill_value = parse_parameter(float, fill_value, "Fill Value")

    output_df = handle_missing_replace_missing_values(
        df, input_column, output_column, fill_value, df.schema[input_column].dataType
    )

    return default_spark(output_df)


def handle_missing_add_indicator_for_missing(df, input_column=None, output_column=None, trained_parameters=None):
    expects_column(df, input_column, "Input column")
    indicator = handle_missing_get_indicator_column(df, input_column, df.schema[input_column].dataType)
    output_column = f"{input_column}_indicator" if not output_column else output_column
    df = df.withColumn(output_column, indicator)

    return default_spark(df)


def handle_missing_drop_rows(df, input_column=None, dimension=None, drop_rows_parameters=None, trained_parameters=None):
    """
    dimension and drop_rows_parameters are the old interface, we keep them from backward compatibility
    input_column is the new interface
    """
    if dimension:
        # old interface is used - convert to new interface
        assert dimension == "Drop Rows"
        input_column = drop_rows_parameters["input_column"]

    indicator_col_name = temp_col_name(df)
    if input_column:
        expects_column(df, input_column, "Input column")
        indicator = handle_missing_get_indicator_column(df, input_column, df.schema[input_column].dataType)
        output_df = df.withColumn(indicator_col_name, indicator)
    else:
        output_df = df
        for f in df.schema.fields:
            indicator = handle_missing_get_indicator_column(df, "`" + f.name + "`", f.dataType)
            if indicator_col_name in output_df.columns:
                output_df = output_df.withColumn(
                    indicator_col_name, sf.when(indicator | output_df[indicator_col_name], True).otherwise(False)
                )
            else:
                output_df = df.withColumn(indicator_col_name, indicator)
    output_df = output_df.where(f"{indicator_col_name} == 0").drop(indicator_col_name)
    return default_spark(output_df)


from pyspark.sql.types import NumericType
import pyspark.sql.functions as sf
import numpy as np


def handle_outliers_standard_deviation_detector(column, df, standard_deviations=None):
    df = df.filter(sf.col(column).isNotNull() & ~sf.col(column).isin(float("nan"), float("inf"), -float("inf")))
    df_stats = df.select(sf.mean(sf.col(column)).alias("mean"), sf.stddev(sf.col(column)).alias("std")).collect()
    mean = df_stats[0]["mean"]
    stddev = df_stats[0]["std"]
    if mean is None or stddev is None:
        return 0, 0
    STD = parse_parameter(float, standard_deviations, "standard_deviations", 4.0)
    return mean - STD * stddev, mean + STD * stddev


def handle_outliers_robust_standard_deviation_detector(
    column, df, lower_quantile, upper_quantile, standard_deviations=None
):
    lower_quantile = parse_parameter(float, lower_quantile, "lower_quantile", 0.25)
    upper_quantile = parse_parameter(float, upper_quantile, "upper_quantile", 0.75)
    df = df.filter(sf.col(column).isNotNull())
    quantiles = df.approxQuantile(column, [lower_quantile, upper_quantile], relativeError=0.001)
    if len(quantiles) < 2:
        return 0, 0
    df = df.filter((sf.col(column) >= quantiles[0]) & (sf.col(column) <= quantiles[1]))
    return handle_outliers_standard_deviation_detector(column, df, standard_deviations)


def handle_outliers_quantile_detector(column, df, lower_quantile, upper_quantile):
    lower_quantile = parse_parameter(float, lower_quantile, "lower_quantile", 0.01)
    upper_quantile = parse_parameter(float, upper_quantile, "upper_quantile", 0.99)
    df = df.filter(sf.col(column).isNotNull())
    quantiles = df.approxQuantile(column, [lower_quantile, upper_quantile], relativeError=0.001)
    if len(quantiles) < 2:
        return 0, 0
    return quantiles[0], quantiles[1]


def handle_outliers_fix_outliers_clip(df, input_column, output_column, lower, upper):
    return df.withColumn(
        output_column,
        sf.when(sf.isnan(df[input_column]), float("nan"))
        .when(df[input_column] < lower, lower)
        .when(df[input_column] > upper, upper)
        .otherwise(df[input_column])
        .cast("double"),
    )


def handle_outliers_fix_outliers_invalidate(df, input_column, output_column, lower, upper):
    return df.withColumn(
        output_column,
        sf.when(sf.isnan(df[input_column]) | (df[input_column] < lower) | (df[input_column] > upper), None).otherwise(
            df[input_column]
        ),
    )


def handle_outliers_fix_outliers_remove(df, input_column, output_column, lower, upper):
    df = df.filter((sf.col(input_column) >= lower) & (sf.col(input_column) <= upper))
    df = df.withColumn(output_column, sf.col(input_column))
    return df


HANDLE_OUTLIERS_FIX_METHOD_CLIP = "Clip"
HANDLE_OUTLIERS_FIX_METHOD_REMOVE = "Remove"
HANDLE_OUTLIERS_FIX_METHOD_INVALIDATE = "Invalidate"


HANDLE_OUTLIERS_FIX_METHOD_MAP = {
    HANDLE_OUTLIERS_FIX_METHOD_CLIP: handle_outliers_fix_outliers_clip,
    HANDLE_OUTLIERS_FIX_METHOD_REMOVE: handle_outliers_fix_outliers_remove,
    HANDLE_OUTLIERS_FIX_METHOD_INVALIDATE: handle_outliers_fix_outliers_invalidate,
}


def handle_outliers_fix_outliers(df, input_column, output_column, fix_method, lower_threshold, upper_threshold):
    if fix_method is None:
        fix_method = HANDLE_OUTLIERS_FIX_METHOD_CLIP
    fixer = HANDLE_OUTLIERS_FIX_METHOD_MAP.get(fix_method)
    if fixer is None:
        raise OperatorCustomerError(f"Invalid fix method selected: {fix_method}")
    return fixer(df, input_column, output_column, lower_threshold, upper_threshold)


def handle_outliers_standard_deviation_numeric_outliers(
    df, input_column=None, output_column=None, fix_method=None, standard_deviations=None, trained_parameters=None
):
    expects_column(df, input_column, "Input column")
    output_column = output_column if output_column else input_column
    if not isinstance(df.schema[input_column].dataType, NumericType):
        raise OperatorCustomerError(
            f"Only numeric columns are supported. Column '{input_column}' has type {df.schema[input_column].dataType}. "
            f"Please cast to a numeric type first."
        )

    if standard_deviations < 0:
        raise OperatorCustomerError("Standard deviations must be a positive number.")

    trained_parameters = load_trained_parameters(
        trained_parameters, {"standard_deviations": standard_deviations, "fix_method": fix_method}
    )

    lower_threshold = parse_parameter(
        float, trained_parameters.get("lower_threshold"), "Trained parameters", nullable=True
    )
    upper_threshold = parse_parameter(
        float, trained_parameters.get("upper_threshold"), "Trained parameters", nullable=True
    )

    if lower_threshold is None or upper_threshold is None:
        lower_threshold, upper_threshold = handle_outliers_standard_deviation_detector(
            input_column, df, standard_deviations
        )
        trained_parameters["lower_threshold"] = lower_threshold
        trained_parameters["upper_threshold"] = upper_threshold

    df = handle_outliers_fix_outliers(df, input_column, output_column, fix_method, lower_threshold, upper_threshold)

    return default_spark_with_trained_parameters(df, trained_parameters)


def handle_outliers_robust_standard_deviation_numeric_outliers(
    df,
    input_column=None,
    output_column=None,
    fix_method=None,
    upper_quantile=None,
    lower_quantile=None,
    standard_deviations=None,
    trained_parameters=None,
):
    expects_column(df, input_column, "Input column")
    output_column = output_column if output_column else input_column
    if not isinstance(df.schema[input_column].dataType, NumericType):
        raise OperatorCustomerError(
            f"Only numeric columns are supported. Column '{input_column}' has type {df.schema[input_column].dataType}. "
            f"Please cast to a numeric type first."
        )

    lower_quantile = parse_parameter(float, lower_quantile, "lower_quantile", 0.25)
    upper_quantile = parse_parameter(float, upper_quantile, "upper_quantile", 0.75)

    if lower_quantile < 0 or lower_quantile > 1.0:
        raise OperatorCustomerError("Lower quantile must be a positive number between 0 and 1.")

    if upper_quantile < 0 or upper_quantile > 1.0:
        raise OperatorCustomerError("Upper quantile must be a positive number between 0 and 1.")

    trained_parameters = load_trained_parameters(
        trained_parameters,
        {
            "fix_method": fix_method,
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "standard_deviations": standard_deviations,
        },
    )

    lower_threshold = parse_parameter(
        float, trained_parameters.get("lower_threshold"), "Trained parameters", nullable=True
    )
    upper_threshold = parse_parameter(
        float, trained_parameters.get("upper_threshold"), "Trained parameters", nullable=True
    )

    if lower_threshold is None or upper_threshold is None:
        lower_threshold, upper_threshold = handle_outliers_robust_standard_deviation_detector(
            input_column, df, lower_quantile, upper_quantile, standard_deviations
        )
        trained_parameters["lower_threshold"] = lower_threshold
        trained_parameters["upper_threshold"] = upper_threshold

    df = handle_outliers_fix_outliers(df, input_column, output_column, fix_method, lower_threshold, upper_threshold)

    return default_spark_with_trained_parameters(df, trained_parameters)


def handle_outliers_quantile_numeric_outliers(
    df,
    input_column=None,
    output_column=None,
    fix_method=None,
    upper_quantile=None,
    lower_quantile=None,
    trained_parameters=None,
):
    expects_column(df, input_column, "Input column")
    output_column = output_column if output_column else input_column
    if not isinstance(df.schema[input_column].dataType, NumericType):
        raise OperatorCustomerError(
            f"Only numeric columns are supported. Column '{input_column}' has type {df.schema[input_column].dataType}. "
            f"Please cast to a numeric type first."
        )

    lower_quantile = parse_parameter(float, lower_quantile, "lower_quantile", 0.01)
    upper_quantile = parse_parameter(float, upper_quantile, "upper_quantile", 0.99)

    if lower_quantile < 0 or lower_quantile > 1.0:
        raise OperatorCustomerError("Lower quantile must be a positive number between 0 and 1.")

    if upper_quantile < 0 or upper_quantile > 1.0:
        raise OperatorCustomerError("Upper quantile must be a positive number between 0 and 1.")

    trained_parameters = load_trained_parameters(
        trained_parameters,
        {"fix_method": fix_method, "lower_quantile": lower_quantile, "upper_quantile": upper_quantile},
    )

    lower_threshold = parse_parameter(
        float, trained_parameters.get("lower_threshold"), "Trained parameters", nullable=True
    )
    upper_threshold = parse_parameter(
        float, trained_parameters.get("upper_threshold"), "Trained parameters", nullable=True
    )

    if lower_threshold is None or upper_threshold is None:
        lower_threshold, upper_threshold = handle_outliers_quantile_detector(
            input_column, df, lower_quantile, upper_quantile
        )
        trained_parameters["lower_threshold"] = lower_threshold
        trained_parameters["upper_threshold"] = upper_threshold

    df = handle_outliers_fix_outliers(df, input_column, output_column, fix_method, lower_threshold, upper_threshold)

    return default_spark_with_trained_parameters(df, trained_parameters)


def handle_outliers_min_max_numeric_outliers(
    df,
    input_column=None,
    output_column=None,
    fix_method=None,
    upper_threshold=None,
    lower_threshold=None,
    trained_parameters=None,
):
    expects_column(df, input_column, "Input column")
    output_column = output_column if output_column else input_column
    if not isinstance(df.schema[input_column].dataType, NumericType):
        raise OperatorCustomerError(
            f"Only numeric columns are supported. Column '{input_column}' has type {df.schema[input_column].dataType}. Please cast to a numeric type first."
        )

    lower_threshold = parse_parameter(float, lower_threshold, "lower_threshold", None)
    if lower_threshold is None:
        raise OperatorCustomerError("For user defined outliers detector, lower_threshold must be provided")

    upper_threshold = parse_parameter(float, upper_threshold, "upper_threshold", None)
    if upper_threshold is None:
        raise OperatorCustomerError("For user defined outliers detector, upper_threshold must be provided")

    df = handle_outliers_fix_outliers(df, input_column, output_column, fix_method, lower_threshold, upper_threshold)
    return default_spark_with_trained_parameters(df, trained_parameters)


def handle_outliers_replace_rare_values(
    df,
    input_column=None,
    output_column=None,
    replacement_string=None,
    absolute_threshold=None,
    fraction_threshold=None,
    max_common_categories=None,
    trained_parameters=None,
):
    """ Replace rare categories with a special token.
        Args:
            df: input pyspark dataframe
            spark: spark session
            column: name of input column
            output_column: name of output column
            operator_params: a dictionary with the following
                rare_string: rare categories will be converted to this new value
                absolute_threshold: An absolute threshold (exclusive) for the minimum number of appearances of a
                category to be classified as not rare. Any category with num_appearances <= absolute_threshold is
                considered rare.
                fraction_threshold: A relative threshold for the minimum number of appearances of a category to be
                    considered not rare. Any category with num_appearances <= fraction_threshold * num_rows is
                    considered rare.
                max_common_categories: Maximum number of categories to be kept as not rare. If the threshold do not
                    filter enough categories, those with the top number of appearances will be classified as not
                    rare.
        Returns:
            df: pyspark dataframe identical to the input df with the exception of the input column having the values
                of the rare categories replaced.
        Note: The new value for rare categories will be of type text.
    """
    expects_column(df, input_column, "Input column")

    # process inputs
    rare_string = replacement_string
    absolute_threshold = parse_parameter(int, absolute_threshold, "absolute_threshold", 10)
    fraction_threshold = parse_parameter(float, fraction_threshold, "fraction_threshold", 0.005)
    max_common_categories = parse_parameter(int, max_common_categories, "max_common_categories", 0)
    output_column = output_column if output_column else input_column

    if absolute_threshold < 0 or np.isnan(absolute_threshold) or np.isinf(absolute_threshold):
        raise OperatorCustomerError(
            f"Incorrect input parameter. absolute_threshold={absolute_threshold} expected to be non-negative"
        )
    if fraction_threshold < 0 or fraction_threshold > 1 or np.isnan(fraction_threshold) or np.isinf(fraction_threshold):
        raise OperatorCustomerError(
            "Incorrect input parameter. " f"fraction_threshold expected to be in [0,1]. received {fraction_threshold}"
        )
    if max_common_categories < 0 or np.isnan(max_common_categories) or np.isinf(max_common_categories):
        raise OperatorCustomerError(
            "Incorrect input parameter. max_common_categories "
            f"expected to be non-negative. received {max_common_categories}"
        )
    elif max_common_categories > 100:
        raise OperatorCustomerError(
            f"Maximum number of common categories supported is 100. Received: {max_common_categories}"
        )

    trained_parameters = load_trained_parameters(
        trained_parameters,
        {
            "replacement_string": replacement_string,
            "absolute_threshold": absolute_threshold,
            "fraction_threshold": fraction_threshold,
            "max_common_categories": max_common_categories,
        },
    )

    common_categories = parse_parameter(
        list, trained_parameters.get("common_categories"), "Trained parameters", nullable=True
    )
    if common_categories is not None:
        common_categories = set(common_categories)
    else:
        # compute an absolute threshold based on the fraction and absolute given
        threshold = max(fraction_threshold * df.count(), absolute_threshold)

        # edge case when the column name is 'count'. If it happens, rename it before using groupBy
        if input_column == "count":
            col_for_group = temp_col_name(df)
            df_to_group = df.withColumn(col_for_group, sf.col(input_column))
        else:
            df_to_group = df
            col_for_group = input_column

        # count for every category how many times it appears.
        category_counts = df_to_group.select(col_for_group).groupBy(col_for_group).count()
        # Filter out those that don't appear sufficiently many times, and sort the remaining by their frequency
        category_counts = category_counts.filter(category_counts["count"] > threshold).sort("count", ascending=False)
        # we have a limit of at most max_common_categories common categories. Here we make sure that this limit is
        # respected.
        if max_common_categories > 0:
            common_categories = category_counts.take(max_common_categories)
        else:
            common_categories = category_counts.collect()
        # convert the list of Row objects to a set of categories
        common_categories = {x[col_for_group] for x in common_categories}
        trained_parameters["common_categories"] = list(common_categories)

    # use the set of categories to convert rare categories into the provided rare_string
    output_df = df.withColumn(
        output_column, sf.when(df[input_column].isin(common_categories), df[input_column]).otherwise(rare_string)
    )

    return default_spark_with_trained_parameters(output_df, trained_parameters)


from typing import List, Union



def manage_columns_drop_column(df, column_to_drop=None, trained_parameters=None):
    column_to_drop = column_to_drop if isinstance(column_to_drop, list) else [column_to_drop]
    expects_column(df, column_to_drop, "Column to drop")
    output_df = df.drop(*column_to_drop)
    return default_spark(output_df)


def manage_columns_duplicate_column(df, input_column=None, new_name=None, trained_parameters=None):
    expects_column(df, input_column, "Input column")
    expects_valid_column_name(new_name, "New name", orig_column=input_column)
    df = df.withColumn(new_name, df[input_column])
    return default_spark(df)


def manage_columns_rename_column(df, input_column=None, new_name=None, columns_to_rename=None, trained_parameters=None):
    prev_single_col_params = [{"input_column": input_column, "new_name": new_name}]
    if not columns_to_rename:
        columns_to_rename = prev_single_col_params
    elif input_column and new_name:
        columns_to_rename.extend(prev_single_col_params)

    for col_to_rename in columns_to_rename:
        input_column, new_name = col_to_rename["input_column"], col_to_rename["new_name"]
        expects_column(df, input_column, "Input column")
        expects_valid_column_name(new_name, "New name", orig_column=input_column)
        df = df.withColumnRenamed(input_column, new_name)
    return default_spark(df)


def validate_column_with_ordering(df, columns_to_move: Union[List, str]):
    # Maintain backwards compatability with single column_to_move as a string
    columns_to_move = columns_to_move if isinstance(columns_to_move, list) else [columns_to_move]
    expects_column(df, columns_to_move, "Column to move")
    # Preserve ordering of columns when moved
    columns_to_move = sorted(columns_to_move, key=df.columns.index)
    return columns_to_move


def get_target_column_index(df, column_to_move: Union[List, str], target_column: str) -> int:
    # checks if target column is valid
    columns_to_move = column_to_move if isinstance(column_to_move, list) else [column_to_move]
    expects_column(df, target_column, "Target column")
    expect_valid_column_to_move(columns_to_move, target_column)

    column_to_move_set = set(columns_to_move)
    columns_without_move_column = [col for col in df.columns if col not in column_to_move_set]
    target_index = columns_without_move_column.index(target_column)
    return target_index


def manage_columns_move_to_index(df, column_to_move: Union[List, str] = None, index=None, trained_parameters=None):
    index = parse_parameter(int, index, "Index")
    column_to_move = validate_column_with_ordering(df, column_to_move)
    column_to_move_set = set(column_to_move)
    if index >= len(df.columns) or index < 0:
        raise OperatorCustomerError(
            "Specified index must be less than the number of columns and greater than or equal to zero. Update the index"
            "accordingly and try again."
        )
    columns_without_move_column = [col for col in df.columns if col not in column_to_move_set]
    reordered_columns = columns_without_move_column[:index] + column_to_move + columns_without_move_column[index:]
    df = df.select(escape_column_names(reordered_columns))
    return default_spark(df)


def manage_columns_move_to_start(df, column_to_move: Union[List, str] = None, trained_parameters=None):
    return manage_columns_move_to_index(df, column_to_move, 0)


def manage_columns_move_to_end(df, column_to_move: Union[List, str] = None, trained_parameters=None):
    return manage_columns_move_to_index(df, column_to_move, len(df.columns) - 1)


def manage_columns_move_after(
    df, column_to_move: Union[List, str] = None, target_column: str = None, trained_parameters=None
):
    target_index = get_target_column_index(df, column_to_move, target_column)
    return manage_columns_move_to_index(df, column_to_move, target_index + 1)


def manage_columns_move_before(
    df, column_to_move: Union[List, str] = None, target_column: str = None, trained_parameters=None
):
    target_index = get_target_column_index(df, column_to_move, target_column)
    return manage_columns_move_to_index(df, column_to_move, target_index)


def manage_columns_move_column(df, **kwargs):
    return dispatch(
        "move_type",
        [df],
        kwargs,
        {
            "Move to start": (manage_columns_move_to_start, "move_to_start_parameters"),
            "Move to end": (manage_columns_move_to_end, "move_to_end_parameters"),
            "Move to index": (manage_columns_move_to_index, "move_to_index_parameters"),
            "Move after": (manage_columns_move_after, "move_after_parameters"),
            "Move before": (manage_columns_move_before, "move_before_parameters"),
        },
    )


def manage_columns_concat_columns(df, left_column, right_column, delimiter="_", output_column=None, **kwargs):
    """
    Transformer to concatenate two columns.

    Parameters
    ----------
    left_column: str, name of column to have as beginning of concatenated column
    right_column: str, name of column to have as end of concatenated column
    delimiter: str, string to place between values from each column. Default is "_"
    output_column: str, name of column to place concatenated result. Default is <left_column>_<right_column>

    Returns
    -------
    default_spark(df): dict containing spark dataframe with concatenated column
    """
    # make sure both columns are present in the df
    validate_col_name_in_df(left_column, df.columns)
    validate_col_name_in_df(right_column, df.columns)

    # set output column
    output_column = f"{left_column}_{right_column}" if output_column is None else output_column

    # cast output column to string if it's already in df
    if output_column in df.columns:
        df = df.withColumn(output_column, df[output_column].cast(StringType()))

    # concatenate columns
    df = df.withColumn(
        output_column, sf.concat_ws(delimiter, df[left_column].cast(StringType()), df[right_column].cast(StringType()))
    )

    return default_spark(df)


from pyspark.sql import functions as sf
from pyspark.sql.functions import asc, desc


def manage_rows_shuffle(df, trained_parameters=None):
    return default_spark(df.sort(sf.rand()))


def manage_rows_sort(df, sort_by=None, order=None, conditions=None, trained_parameters=None):
    """
    Sort rows on selected columns in specified orders. If both sort_by and order is provided, conditions is ignored.

    Args:
        df: Spark DataFrame
        sort_by: a single column name to sort the df on
        order: the order that the df should be sorted in, either Ascending or Descending
        conditions: list of sort_by-order pairs to sort the df on multiple columns
    """
    ORDER_FUNCS = {"Ascending": asc, "Descending": desc}

    if not conditions:
        if sort_by is None or order is None:
            raise OperatorCustomerError(
                "You haven't specified a column to sort the data. Specify a column and the sort order and try again."
            )
        conditions = [{"sort_by": sort_by, "order": order}]

    seen = set()
    cols = list()
    for order_pair in conditions:
        sort_by, order = order_pair["sort_by"], order_pair["order"]
        expects_column(df, sort_by, "Sort by column")
        expects_parameter_value_in_list("Order", order, ORDER_FUNCS.keys())
        if sort_by in seen:
            raise OperatorCustomerError(
                f"The parameter value that you've specified is invalid. There are duplicate column names {sort_by}. Remove the duplicate columns and try again."
            )
        seen.add(sort_by)
        cols.append(ORDER_FUNCS[order](sort_by))

    return default_spark(df.sort(cols))


def manage_rows_drop_duplicates(df, trained_parameters=None):
    return default_spark(df.dropDuplicates())


def manage_rows_drop_empty(df, trained_parameters=None):
    return default_spark(df.dropna(how="all"))




class NonCastableDataHandlingMethod(Enum):
    REPLACE_WITH_NULL = "replace_null"
    REPLACE_WITH_NULL_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN = "replace_null_with_new_col"
    REPLACE_WITH_FIXED_VALUE = "replace_value"
    REPLACE_WITH_FIXED_VALUE_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN = "replace_value_with_new_col"
    DROP_NON_CASTABLE_ROW = "drop"

    @staticmethod
    def get_names():
        return [item.name for item in NonCastableDataHandlingMethod]

    @staticmethod
    def get_values():
        return [item.value for item in NonCastableDataHandlingMethod]


class MohaveDataType(Enum):
    BOOL = "bool"
    DATE = "date"
    DATETIME = "datetime"
    FLOAT = "float"
    LONG = "long"
    STRING = "string"
    ARRAY = "array"
    STRUCT = "struct"
    OBJECT = "object"
    IMAGE = "image"

    @staticmethod
    def get_names():
        return [item.name for item in MohaveDataType]

    @staticmethod
    def get_values():
        return [item.value for item in MohaveDataType]


PYTHON_TYPE_MAPPING = {
    MohaveDataType.BOOL: bool,
    MohaveDataType.DATE: str,
    MohaveDataType.DATETIME: str,
    MohaveDataType.FLOAT: float,
    MohaveDataType.LONG: int,
    MohaveDataType.STRING: str,
    MohaveDataType.ARRAY: str,
    MohaveDataType.STRUCT: str,
}

MOHAVE_TO_SPARK_TYPE_MAPPING = {
    MohaveDataType.BOOL: BooleanType,
    MohaveDataType.DATE: DateType,
    MohaveDataType.DATETIME: TimestampType,
    MohaveDataType.FLOAT: DoubleType,
    MohaveDataType.LONG: LongType,
    MohaveDataType.STRING: StringType,
    MohaveDataType.ARRAY: ArrayType,
    MohaveDataType.STRUCT: StructType,
}

SPARK_TYPE_MAPPING_TO_SQL_TYPE = {
    BooleanType: "BOOLEAN",
    LongType: "BIGINT",
    DoubleType: "DOUBLE",
    StringType: "STRING",
    DateType: "DATE",
    TimestampType: "TIMESTAMP",
}

SPARK_TO_MOHAVE_TYPE_MAPPING = {value: key for (key, value) in MOHAVE_TO_SPARK_TYPE_MAPPING.items()}


def cast_column_helper(df, column, mohave_data_type, date_col, datetime_col, non_date_col):
    """Helper for casting a single column to a data type."""
    if mohave_data_type == MohaveDataType.DATE:
        return df.withColumn(column, date_col)
    elif mohave_data_type == MohaveDataType.DATETIME:
        return df.withColumn(column, datetime_col)
    else:
        return df.withColumn(column, non_date_col)


def cast_single_column_type(
    df,
    column,
    mohave_data_type,
    invalid_data_handling_method,
    replace_value=None,
    date_formatting="dd-MM-yyyy",
    datetime_formatting=None,
):
    """Cast single column to a new type

    Args:
        df (DataFrame): spark dataframe
        column (Column): target column for type casting
        mohave_data_type (Enum): Enum MohaveDataType
        invalid_data_handling_method (Enum): Enum NonCastableDataHandlingMethod
        replace_value (str): value to replace for invalid data when "replace_value" is specified
        date_formatting (str): format for date. Default format is "dd-MM-yyyy"
        datetime_formatting (str): format for datetime. Default is None, indicates auto-detection

    Returns:
        df (DataFrame): casted spark dataframe
    """
    cast_to_date = sf.to_date(df[column], date_formatting)
    to_ts = sf.pandas_udf(f=to_timestamp_single, returnType="string")
    if datetime_formatting is None:
        cast_to_datetime = sf.to_timestamp(to_ts(df[column]))  # auto-detect formatting
    else:
        cast_to_datetime = sf.to_timestamp(df[column], datetime_formatting)
    cast_to_non_date = df[column].cast(MOHAVE_TO_SPARK_TYPE_MAPPING[mohave_data_type]())
    non_castable_column = f"{column}_typecast_error"
    temp_column = "temp_column"

    if invalid_data_handling_method == NonCastableDataHandlingMethod.REPLACE_WITH_NULL:
        # Replace non-castable data to None in the same column. pyspark's default behaviour
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+---+--+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long
        # +---+------+
        # | id | txt |
        # +---+------+
        # | 1 | None |
        # | 2 | None |
        # | 3 | 1    |
        # +---+------+
        return cast_column_helper(
            df,
            column,
            mohave_data_type,
            date_col=cast_to_date,
            datetime_col=cast_to_datetime,
            non_date_col=cast_to_non_date,
        )
    if invalid_data_handling_method == NonCastableDataHandlingMethod.DROP_NON_CASTABLE_ROW:
        # Drop non-castable row
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+---+--+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long, _ non-castable row
        # +---+----+
        # | id|txt |
        # +---+----+
        # |  3|  1 |
        # +---+----+
        df = cast_column_helper(
            df,
            column,
            mohave_data_type,
            date_col=cast_to_date,
            datetime_col=cast_to_datetime,
            non_date_col=cast_to_non_date,
        )
        return df.where(df[column].isNotNull())

    if (
        invalid_data_handling_method
        == NonCastableDataHandlingMethod.REPLACE_WITH_NULL_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN
    ):
        # Replace non-castable data to None in the same column and put non-castable data to a new column
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+------+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long
        # +---+----+------------------+
        # | id|txt |txt_typecast_error|
        # +---+----+------------------+
        # |  1|None|      foo         |
        # |  2|None|      bar         |
        # |  3|  1 |                  |
        # +---+----+------------------+
        df = cast_column_helper(
            df,
            temp_column,
            mohave_data_type,
            date_col=cast_to_date,
            datetime_col=cast_to_datetime,
            non_date_col=cast_to_non_date,
        )
        df = df.withColumn(non_castable_column, sf.when(df[temp_column].isNotNull(), "").otherwise(df[column]),)
    elif invalid_data_handling_method == NonCastableDataHandlingMethod.REPLACE_WITH_FIXED_VALUE:
        # Replace non-castable data to a value in the same column
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+------+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long, replace non-castable value to 0
        # +---+-----+
        # | id| txt |
        # +---+-----+
        # |  1|  0  |
        # |  2|  0  |
        # |  3|  1  |
        # +---+----+
        value = _validate_and_cast_value(value=replace_value, mohave_data_type=mohave_data_type)

        df = cast_column_helper(
            df,
            temp_column,
            mohave_data_type,
            date_col=cast_to_date,
            datetime_col=cast_to_datetime,
            non_date_col=cast_to_non_date,
        )

        replace_date_value = sf.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(
            sf.to_date(sf.lit(value), date_formatting)
        )
        replace_non_date_value = sf.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(value)

        df = df.withColumn(
            temp_column, replace_date_value if (mohave_data_type == MohaveDataType.DATE) else replace_non_date_value
        )
    elif (
        invalid_data_handling_method
        == NonCastableDataHandlingMethod.REPLACE_WITH_FIXED_VALUE_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN
    ):
        # Replace non-castable data to a value in the same column and put non-castable data to a new column
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+---+--+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long, replace non-castable value to 0
        # +---+----+------------------+
        # | id|txt |txt_typecast_error|
        # +---+----+------------------+
        # |  1|  0  |   foo           |
        # |  2|  0  |   bar           |
        # |  3|  1  |                 |
        # +---+----+------------------+
        value = _validate_and_cast_value(value=replace_value, mohave_data_type=mohave_data_type)

        df = cast_column_helper(
            df,
            temp_column,
            mohave_data_type,
            date_col=cast_to_date,
            datetime_col=cast_to_datetime,
            non_date_col=cast_to_non_date,
        )
        df = df.withColumn(non_castable_column, sf.when(df[temp_column].isNotNull(), "").otherwise(df[column]),)

        replace_date_value = sf.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(
            sf.to_date(sf.lit(value), date_formatting)
        )
        replace_non_date_value = sf.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(value)

        df = df.withColumn(
            temp_column, replace_date_value if (mohave_data_type == MohaveDataType.DATE) else replace_non_date_value
        )
    # drop temporary column
    df = df.withColumn(column, df[temp_column]).drop(temp_column)

    df_cols = df.columns
    if non_castable_column in df_cols:
        # Arrange columns so that non_castable_column col is next to casted column
        df_cols.remove(non_castable_column)
        column_index = df_cols.index(column)
        arranged_cols = df_cols[: column_index + 1] + [non_castable_column] + df_cols[column_index + 1 :]
        df = df.select(*arranged_cols)
    return df


def _validate_and_cast_value(value, mohave_data_type):
    if value is None:
        return value
    try:
        return PYTHON_TYPE_MAPPING[mohave_data_type](value)
    except ValueError as e:
        raise ValueError(
            f"Invalid value to replace non-castable data. "
            f"{mohave_data_type} is not in mohave supported date type: {MohaveDataType.get_values()}. "
            f"Please use a supported type",
            e,
        )





DEFAULT_TIMESTAMP_FORMAT_INFERENCE_SAMPLE_SIZE = 5000

# This list contains the supported timestamp formats for Spark cast to timestamp function.
# See https://spark.apache.org/docs/latest/sql-ref-datetime-pattern.html.
# In the list, patterns should be ordered so that each pattern string before all of its
# prefixes (e.g. "yyyy-mm-dd'T'HH:mm:ss" before "yyyy-mm-dd").
# This is because Spark's to_timestamp() function throws an error if a pattern only matches prefix of the timestamp
# string (e.g. "yyyy-mm-dd" for "2019-01-01T12:00:00").
# In spark_cast_datetime_column(), the early match with more specific pattern could prevent potential error being raised
# by the later prefix match with less specific pattern.
SPARK_SUPPORTED_TIMESTAMP_FORMATS_HYPHEN = [
    "yyyy-MM-dd HH:mm:ss.SSSXXX",
    "yyyy-MM-dd HH:mm:ss.SSSXX",
    "yyyy-MM-dd HH:mm:ss.SSSX",
    "yyyy-MM-dd HH:mm:ss.SSS",
    "yyyy-MM-dd HH:mm:ssXXX",
    "yyyy-MM-dd HH:mm:ssXX",
    "yyyy-MM-dd HH:mm:ssX",
    "yyyy-MM-dd HH:mm:ss",
    "yyyy-MM-dd HH:mm",
    "yyyy-MM-dd",
]
SPARK_SUPPORTED_TIMESTAMP_FORMATS_HYPHEN_T = [
    "yyyy-MM-dd'T'HH:mm:ss.SSSXXX",
    "yyyy-MM-dd'T'HH:mm:ss.SSSXX",
    "yyyy-MM-dd'T'HH:mm:ss.SSSX",
    "yyyy-MM-dd'T'HH:mm:ss.SSS",
    "yyyy-MM-dd'T'HH:mm:ssXXX",
    "yyyy-MM-dd'T'HH:mm:ssXX",
    "yyyy-MM-dd'T'HH:mm:ssX",
    "yyyy-MM-dd'T'HH:mm:ss",
    "yyyy-MM-dd'T'HH:mm",
]
SPARK_SUPPORTED_TIMESTAMP_FORMATS_SLASH = [
    "MM/dd/yyyy HH:mm:ss.SSSXXX",
    "MM/dd/yyyy HH:mm:ss.SSSXX",
    "MM/dd/yyyy HH:mm:ss.SSSX",
    "MM/dd/yyyy HH:mm:ss.SSS",
    "MM/dd/yyyy HH:mm:ssX",
    "MM/dd/yyyy HH:mm:ss",
    "MM/dd/yyyy HH:mm",
    "MM/dd/yyyy",
]
SPARK_SUPPORTED_TIMESTAMP_FORMATS_OTHERS = [
    "yyyyMMdd",
]


def _get_spark_supported_timestamp_format_group(datetime_string):
    """
    Get a SPARK_SUPPORTED_TIMESTAMP_FORMAT array based on the special character(s) in a datetime string.
    It helps reduce the _guess_spark_datetime_format() overhead by parsing with a smaller set of formats.
    Note: The if else branch order matters.

    Parameters
    ----------
    datetime_string: str
        a datetime string to guess the format of

    Returns
    ----------
    list: a SPARK_SUPPORTED_TIMESTAMP_FORMATS_XXX list
    """

    if "/" in datetime_string:
        return SPARK_SUPPORTED_TIMESTAMP_FORMATS_SLASH
    elif "T" in datetime_string and "-" in datetime_string:
        return SPARK_SUPPORTED_TIMESTAMP_FORMATS_HYPHEN_T
    elif "-" in datetime_string:
        return SPARK_SUPPORTED_TIMESTAMP_FORMATS_HYPHEN
    return SPARK_SUPPORTED_TIMESTAMP_FORMATS_OTHERS


def to_timestamp_single(x):
    """Helper function for auto-detecting datetime format and casting to ISO-8601 string."""
    converted = pd.to_datetime(x, errors="coerce")
    return converted.astype("str").replace("NaT", "")  # makes pandas NaT into empty string


def _pandas_cast_datetime_column(df, col_name):
    """Cast a DateTime column using Pandas."""
    to_ts = sf.pandas_udf(f=to_timestamp_single, returnType="string")
    return df.withColumn(col_name, sf.to_timestamp(to_ts(df[col_name])))


def _spark_cast_datetime_column(df, col_name, datetime_format):
    """Cast a DateTime column using PySpark."""
    return df.withColumn(col_name, sf.to_timestamp(col_name, datetime_format),)


def _guess_spark_datetime_format(datetime_string, spark):
    """
    Guess the spark datetime format of a given datetime string.

    Parameters
    ----------
    datetime_string: str
        a dateime string to guess the format of.
    spark: spark session

    Returns
    ----------
    str or None:
        a spark timestamp format string
        or None if it can't be guessed.
    """
    if type(datetime_string) is not str:
        logging.info("The datetime_string is not a string: can't guess its spark datetime format.")
        return None

    col_name = "temp_col"
    temp_df = spark.createDataFrame([datetime_string], StringType()).toDF(col_name)

    format_group = _get_spark_supported_timestamp_format_group(datetime_string)
    for timestamp_format in format_group:
        try:
            parsed_df = temp_df.withColumn(col_name, sf.to_timestamp(col_name, timestamp_format))

            if parsed_df.first()[col_name] is not None:
                return timestamp_format
        except Exception:
            logging.info(f"Thrown exception when parsing datetime on format: {timestamp_format}.")


def _guess_spark_datetime_format_for_column(df, col_name, spark):
    """
    Guess a given dataframe column's spark datetime format based on the first non-None value from the top 5000 rows.
    The 5000 rows is the same sample size used for column type inference.

    This function implements a logic same as the pandas's _guess_datetime_format_for_array()
    https://github.com/pandas-dev/pandas/blob/814fd82beadf7155f8429c58c27425edd241922c/pandas/core/tools/datetimes.py#L128

    Parameters
    ----------
    df: spark dataframe
    col_name: str
        the column to infer the timestamp format
    spark: spark session

    Returns
    ----------
    str or None:
        a spark timestamp format string
        or None if it can't be guessed.
    """
    sample_df = df.select(col_name).limit(DEFAULT_TIMESTAMP_FORMAT_INFERENCE_SAMPLE_SIZE)
    first_non_nan_value_df = sample_df.withColumn(col_name, sf.first(col_name, ignorenulls=True),)

    if first_non_nan_value_df.count() == 0:
        logging.info(
            f"All of the top {DEFAULT_TIMESTAMP_FORMAT_INFERENCE_SAMPLE_SIZE} values are None: can't guess this column's datetime format."
        )
        return None

    first_non_nan_value = first_non_nan_value_df.head(1)[0][0]

    return _guess_spark_datetime_format(first_non_nan_value, spark)


def cast_datetime_column(df, col_name, spark):
    """
    Casts DateTime column values with PySpark for performance purpose.
    If the result is difference from pandas cast value, fallback to pandas.

    Parameters
    ----------
    df: spark dataframe
    col_name: str
        name of the DateTime column

    Returns
    ----------
    DataFrame containing the cast column
    """

    if len(df.head(1)) == 0:
        logging.info(f"Empty dataframe: no need to cast column.")
        return df

    guessed_datetime_format = _guess_spark_datetime_format_for_column(df, col_name, spark)
    if guessed_datetime_format:
        logging.info(f"Cast datetime column using a Spark timestamp format: {guessed_datetime_format}.")
        return _spark_cast_datetime_column(df, col_name, guessed_datetime_format)
    else:
        logging.info("Cast datetime column using the Pandas implementation.")
        return _pandas_cast_datetime_column(df, col_name)




class OperatorCustomerError(Exception):
    """Error type for Customer Errors in Spark Operators"""


class OperatorPythonError(OperatorCustomerError):
    """Error type for Python exceptions from UDFs in Spark Operators"""


import shutil




def is_inference_running_mode():
    return False


def temp_col_name(df, *illegal_names, prefix: str = "temp_col"):
    """Generates a temporary column name that is unused.
    """
    name = prefix
    idx = 0
    name_set = set(list(df.columns) + list(illegal_names))
    while name in name_set:
        name = f"_{prefix}_{idx}"
        idx += 1

    return name


def get_temp_col_if_not_set(df, col_name):
    """Extracts the column name from the parameters if it exists, otherwise generates a temporary column name.
    """
    if col_name:
        return col_name, False
    else:
        return temp_col_name(df), True


def replace_input_if_output_is_temp(df, input_column, output_column, output_is_temp):
    """Replaces the input column in the dataframe if the output was not set

    This is used with get_temp_col_if_not_set to enable the behavior where a 
    transformer will replace its input column if an output is not specified.
    """
    if output_is_temp:
        df = df.withColumn(input_column, df[output_column])
        df = df.drop(output_column)
        return df
    else:
        return df


def parse_parameter(typ, value, key, default=None, nullable=False):
    if value is None:
        if default is not None or nullable:
            return default
        else:
            raise OperatorCustomerError(f"Missing required input: '{key}'")
    else:
        try:
            value = typ(value)
            if isinstance(value, (int, float, complex)) and not isinstance(value, bool):
                if np.isnan(value) or np.isinf(value):
                    raise OperatorCustomerError(
                        f"Invalid value provided for '{key}'. Expected {typ.__name__} but received: {value}"
                    )
                else:
                    return value
            else:
                return value
        except (ValueError, TypeError):
            raise OperatorCustomerError(
                f"Invalid value provided for '{key}'. Expected {typ.__name__} but received: {value}"
            )
        except OverflowError:
            raise OperatorCustomerError(
                f"Overflow Error: Invalid value provided for '{key}'. Given value '{value}' exceeds the range of type "
                f"'{typ.__name__}' for this input. Insert a valid value for type '{typ.__name__}' and try your request "
                f"again."
            )


def expects_valid_column_name(value, key, nullable=False, orig_column=None):
    if nullable and value is None:
        return
    if value is None or len(str(value).strip()) == 0:
        raise OperatorCustomerError(
            f"Column name cannot be null, empty, or whitespace for parameter '{key}'. "
            f"Provide a column name with at least one character and try again."
        )

    if orig_column and orig_column == value:
        raise OperatorCustomerError(
            f"The new name ({value}) is the same as the old name ({orig_column}). "
            f"Provide a new column name that is different from the original column name "
            f"and try again."
        )


def expect_valid_column_to_move(columns_to_move: List, target_column: str):
    for column in columns_to_move:
        if column == target_column:
            raise OperatorCustomerError(
                f"Invalid target column name. "
                f"The target column ({target_column}) should not be the same as the column to move {column}. "
                f"Select a different column to be the target column or remove the target column from columns "
                f"you want to move."
            )


def expects_parameter(value, key, condition=None):
    if value is None:
        raise OperatorCustomerError(f"Missing required input: '{key}'")
    elif condition is not None and not condition:
        raise OperatorCustomerError(f"Invalid value provided for '{key}': {value}")


def expects_column(df, columns, key):
    columns = columns if isinstance(columns, list) else [columns]
    for column in columns:
        if df is None or not (isinstance(df, DataFrame) or isinstance(df, pd.DataFrame)):
            raise OperatorCustomerError(
                f"Cannot read from non dataframe object of type {type(df)}. Please verify that the dataframe "
                f"is valid and try again. If the issue persists, contact AWS support."
            )
        if not column or column not in df.columns:
            raise OperatorCustomerError(
                f"The column '{column}' does not exist in your dataset. For '{key}', "
                f"specify a different column name that exists in your dataset and try again."
            )


def expects_parameter_value_in_list(key, value, items):
    if value not in items:
        raise OperatorCustomerError(f"Illegal parameter value. {key} expected to be in {items}, but given {value}")


def expects_parameter_value_in_range(key, value, start, end, nullable=False):
    if nullable and value is None:
        return
    if value is None or (value < start or value > end):
        raise OperatorCustomerError(
            f"Illegal parameter value. {key} expected to be within range {start} - {end}, but given {value}"
        )


def encode_pyspark_model(model, spark):
    with tempfile.TemporaryDirectory() as dirpath:
        dirpath = os.path.join(dirpath, "model")
        model.save(dirpath)


        # Create the temporary zip-file.
        mem_zip = BytesIO()
        with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            # Zip the directory.
            for root, dirs, files in os.walk(dirpath):
                for file in files:
                    rel_dir = os.path.relpath(root, dirpath)
                    zf.write(os.path.join(root, file), os.path.join(rel_dir, file))

        zipped = mem_zip.getvalue()
        encoded = base64.b85encode(zipped)
        return str(encoded, "utf-8")


def decode_pyspark_model(model_factory, encoded):
    with tempfile.TemporaryDirectory() as dirpath:
        zip_bytes = base64.b85decode(encoded)
        mem_zip = BytesIO(zip_bytes)
        mem_zip.seek(0)

        with zipfile.ZipFile(mem_zip, "r") as zf:
            zf.extractall(dirpath)


        model = model_factory.load(dirpath)
        return model


def hash_parameters(value):
    try:
        encoded = json.dumps(value, sort_keys=True).encode(encoding="UTF-8", errors="strict")
        return hashlib.sha1(encoded).hexdigest()
    except:  # noqa: E722
        raise RuntimeError("Object not supported for serialization")


def load_trained_parameters(trained_parameters, operator_parameters):
    trained_parameters = trained_parameters if trained_parameters else {}
    parameters_hash = hash_parameters(operator_parameters)
    stored_hash = trained_parameters.get("_hash")
    if stored_hash != parameters_hash:
        trained_parameters = {"_hash": parameters_hash}
    return trained_parameters


def try_decode_pyspark_model(trained_parameters, model_factory, name):
    try:
        model = decode_pyspark_model(model_factory, trained_parameters[name])
        logging.info(f"Decoded PySpark model {name} from trained_parameters.")
        return model, True
    except Exception as e:
        logging.error(f"Could not decode PySpark model {name} from trained_parameters: {e}")
        del trained_parameters[name]
        return None, False


def load_pyspark_model_from_trained_parameters(trained_parameters, model_factory, name):
    if trained_parameters is None or name not in trained_parameters:
        return None, False

    if is_inference_running_mode():
        if isinstance(trained_parameters[name], str):
            model, model_loaded = try_decode_pyspark_model(trained_parameters, model_factory, name)
            if not model_loaded:
                return model, model_loaded
            trained_parameters[name] = model
        return trained_parameters[name], True

    return try_decode_pyspark_model(trained_parameters, model_factory, name)


def try_decode_pyspark_model_and_store_artifacts(trained_parameters, model_factory, name):
    """ 
    Decode pyspark model that requires model artifacts when transforming the dataframe (e.g., ImputerModel), and store its artifacts in local.
    """
    try:
        model, temp_dir_path = decode_pyspark_model_and_store_artifacts(model_factory, trained_parameters[name])
        logging.info(f"Decoded PySpark model {name} from trained_parameters.")
        return model, True, temp_dir_path
    except Exception as e:
        logging.error(f"Could not decode PySpark model {name} from trained_parameters: {e}")
        del trained_parameters[name]
        return None, False, None


def load_pyspark_model_from_trained_parameters_and_store_artifacts(trained_parameters, model_factory, name):
    """ 
    Load pyspark model that requires model artifacts when transforming the dataframe (e.g., ImputerModel), from trained parameters, and store its artifacts in local.
    """
    if trained_parameters is None or name not in trained_parameters:
        return None, False, None

    temp_dir_path = None
    if is_inference_running_mode():
        if isinstance(trained_parameters[name], str):
            model, model_loaded, temp_dir_path = try_decode_pyspark_model_and_store_artifacts(
                trained_parameters, model_factory, name
            )
            if not model_loaded:
                return model, model_loaded, temp_dir_path
            trained_parameters[name] = model
        return trained_parameters[name], True, temp_dir_path

    return try_decode_pyspark_model_and_store_artifacts(trained_parameters, model_factory, name)


def decode_pyspark_model_and_store_artifacts(model_factory, encoded):
    """ 
    Decode pyspark model that requires model artifacts when transforming the dataframe (e.g., ImputerModel), and store its artifacts in local.
    """
    dirpath = tempfile.mkdtemp()
    zip_bytes = base64.b85decode(encoded)
    mem_zip = BytesIO(zip_bytes)
    mem_zip.seek(0)

    with zipfile.ZipFile(mem_zip, "r") as zf:
        zf.extractall(dirpath)


    model = model_factory.load(dirpath)
    return model, dirpath


def fit_and_save_model(trained_parameters, name, algorithm, df):
    model = algorithm.fit(df)
    spark_session = df.sql_ctx.sparkSession
    trained_parameters[name] = encode_pyspark_model(model, spark_session)
    logging.info(f"Fitted PySpark model {name}")
    return model


def transform_using_trained_model(model, df, loaded):
    try:
        return model.transform(df)
    except Exception as e:
        if loaded:
            raise OperatorCustomerError(
                f"Encountered error while using stored model. Please delete the operator and try again. {e}"
            )
        else:
            raise e


def transform_using_trained_model_and_clean_artifacts(model, df, loaded, temp_dir_path=None):
    """ 
    Transform the dataframe using pyspark model that requires model artifacts when transforming the dataframe (e.g., ImputerModel), and clean its artifacts in local.
    """
    try:
        transformed_df = model.transform(df)
        if temp_dir_path:
            try:
                shutil.rmtree(temp_dir_path)
            except FileNotFoundError:
                logging.info(f"Dir {temp_dir_path} doesn't exist.")
                pass
        return transformed_df
    except Exception as e:
        if loaded:
            raise OperatorCustomerError(
                f"Encountered error while using stored model. Please delete the operator and try again. {e}"
            )
        else:
            raise e


ESCAPE_CHAR_PATTERN = re.compile("[{}]+".format(re.escape(".`")))


def escape_column_name(col):
    """Escape column name so it works properly for Spark SQL"""

    # Do nothing for Column object, which should be already valid/quoted
    if isinstance(col, Column):
        return col

    column_name = col

    if ESCAPE_CHAR_PATTERN.search(column_name):
        column_name = f"`{column_name}`"

    return column_name


def escape_column_names(columns):
    return [escape_column_name(col) for col in columns]


def sanitize_df(df):
    """Sanitize dataframe with Spark safe column names and return column name mappings

    Args:
        df: input dataframe

    Returns:
        a tuple of
            sanitized_df: sanitized dataframe with all Spark safe columns
            sanitized_col_mapping: mapping from original col name to sanitized column name
            reversed_col_mapping: reverse mapping from sanitized column name to original col name
    """

    sanitized_col_mapping = {}
    sanitized_df = df

    for orig_col in df.columns:
        if ESCAPE_CHAR_PATTERN.search(orig_col):
            # create a temp column and store the column name mapping
            temp_col = f"{orig_col.replace('.', '_')}_{temp_col_name(sanitized_df)}"
            sanitized_col_mapping[orig_col] = temp_col

            sanitized_df = sanitized_df.withColumn(temp_col, sanitized_df[f"`{orig_col}`"])
            sanitized_df = sanitized_df.drop(orig_col)

    # create a reversed mapping from sanitized col names to original col names
    reversed_col_mapping = {sanitized_name: orig_name for orig_name, sanitized_name in sanitized_col_mapping.items()}

    return sanitized_df, sanitized_col_mapping, reversed_col_mapping


def add_filename_column(df):
    """Add a column containing the input file name of each record."""
    filename_col_name_prefix = "_data_source_filename"
    filename_col_name = filename_col_name_prefix
    counter = 1
    while filename_col_name in df.columns:
        filename_col_name = f"{filename_col_name_prefix}_{counter}"
        counter += 1
    return df.withColumn(filename_col_name, sf.input_file_name())


class OAuthInvalidGrantError(OperatorCustomerError):
    pass


class OAuthTokenExchangeError(OperatorCustomerError):
    pass


def log_image_custom_code(code: str, custom_type: str):
    IMAGE_IMPORTS = [
        "cv2",
        "imgaug",
        "DEFAULT_IMAGE_COLUMN",
        "IMAGE_COLUMN_TYPE",
        "BasicImageOperationDecorator",
        "PandasUDFOperationDecorator",
    ]

    is_image_code = False
    for _import in IMAGE_IMPORTS:
        if _import in code:
            is_image_code = True
            logging.info(f"Image import detected in custom code: {_import}")

    if is_image_code:
        logging.info(f"Detected image imports in {custom_type} code.")

    return


def should_truncate_image_col(field, selected_columns, fields):
    """
    Check if image column should be truncated. 
    
    If no selected column for preview, check if image is only column in dataframe, if not, truncate column.
    If selected column for preview is not the image column, truncate it.

    Args:
        field (StructField): image column field
        selected_columns (list[str]): selected columns for preview
        fields (list[StructField]): all fields from dataframe schema

    Returns:
        True if image column should be truncated, False otherwise.
    """
    if not OperatorSchemaReader.is_ui_version_compatible(MULTI_MODALITY_UI_VERSION):
        return False

    if selected_columns is None:
        return len(fields) > 1
    else:
        return field.name not in selected_columns


"""Modules with helpers to control Spark partition behavior and number of parts of output files"""

import logging
import sys
from pyspark.sql import DataFrame


def estimate_dataframe_size_bytes(df: DataFrame, num_sample_rows: int = 100) -> int:
    """Estimate dataframe storage size in bytes

    Estimate the size of the dataframe within a small time budgets (30s). The purpose is to quickly infer the size
    to support certain heuristics (auto combine output file parts for small (e.g. <100MB) dataset)

    Returns:
        estimated dataframe size in bytes
    """

    approx_row_count = df.rdd.countApprox(timeout=30)
    num_columns = len(df.columns)

    # We sample the first `num_sample_rows` rows from the data frame and take the average
    # as an estimate for avg per-record size
    sampled_df = df.limit(num_sample_rows).toPandas()
    if len(sampled_df) == 0:
        logging.info(
            f"No records gathered when approximating dataframe size. # rows: {approx_row_count}, # columns: {num_columns}"
        )
        return 0

    # quickly serialize this as csv strings and get the average size
    sample_row_avg_size_in_bytes = len(sampled_df.to_csv()) / len(sampled_df)

    estimated_size_in_bytes = int(approx_row_count * sample_row_avg_size_in_bytes)
    logging.info(
        "Estimated dataframe size info"
        + f"# rows: {approx_row_count}, # columns: {num_columns}, average size per row: {sample_row_avg_size_in_bytes}b, "
        + f"data frame size: {estimated_size_in_bytes} bytes"
    )
    return estimated_size_in_bytes



ACCESSPOINT_ARN_VERBOSE_REGEX = """
            ^(s3://)?arn:(aws|aws-cn):s3:
            (?P<region>.*)  # aws region
            :
            (?P<account_id>\d*)  # aws account id
            :accesspoint/
            (?P<name>[a-zA-Z0-9-]*)  # access point name
            (/(?P<path>.*))?  # optional object path
            """


@dataclass
class SupportedContentType(Enum):
    CSV = "CSV"
    PARQUET = "PARQUET"
    TSV = "CSV"
    ORC = "ORC"
    JSON = "JSON"
    JSONL = "JSONL"
    IMAGE = "IMAGE"


@dataclass
class SupportedImageType(Enum):
    """We support all fully supported formats listed here: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56"""

    BMP = "bmp"
    DIB = "dib"
    HDR = ["pic", "hdr"]
    JPEG = ["jpeg", "jpg", "jpe"]
    JPEG2000 = "jp2"
    PNG = "png"
    PORTABLE = ["pbm", "pgm", "ppm", "pxm", "pnm"]
    RAS = ["sr", "ras"]
    TIFF = ["tiff", "tif"]
    WEBP = "webp"

    @staticmethod
    def list():
        formats = []
        for _format in map(lambda x: x.value, SupportedImageType):
            if isinstance(_format, str):
                formats.append(_format)
            else:
                formats.extend(_format)
        return formats


class S3ObjectType(Enum):
    FILE = "file"
    FOLDER = "folder"


@dataclass
class S3Metadata:
    name: str
    uri: str
    type: str


@dataclass
class S3ObjectMetadata(S3Metadata):
    """A dataclass for modeling a single S3 object metadata.
    """

    content_type: str = None
    size: int = None
    last_modified: str = None


def s3_get_list_objects_response(s3_client, bucket_name, prefix, delimiter, continuation_token=None, max_keys=500):
    request = {
        "Bucket": bucket_name,
        "Delimiter": delimiter,
        "EncodingType": "url",
        "MaxKeys": max_keys,
        "Prefix": prefix,
    }

    if continuation_token:
        request["ContinuationToken"] = continuation_token

    response = s3_client.list_objects_v2(**request)

    return response


def s3_parse_objects(bucket, prefix, response, delimiter, skip_unquote_plus=False):
    objects = []
    if "Contents" not in response:
        return objects
    contents = response["Contents"]
    for obj in contents:
        obj_key = obj["Key"] if skip_unquote_plus else unquote_plus(obj["Key"])
        if (obj_key == prefix or delimiter == "") and not s3_is_file(response, obj_key):
            continue
        obj_name = s3_get_basename(obj_key)
        obj_size = obj["Size"]
        content_type = s3_infer_content_type_v2(uri=obj_key)
        objects.append(
            S3ObjectMetadata(
                name=obj_name,
                uri=s3_format_uri(bucket, obj_key),
                type=S3ObjectType.FILE.value,
                size=obj_size,
                last_modified=str(obj["LastModified"]),
                content_type=content_type,
            )
        )
    return objects


def s3_is_file(response, obj_key):
    try:
        exists = response["CommonPrefixes"]
        return False
    except KeyError:
        if obj_key[-1] == "/":
            return False
        return True


def s3_infer_content_type_v2(uri):
    inferred_content_type: str = PurePath(uri).suffix[1:]
    if inferred_content_type.upper() in SupportedContentType.__members__:
        inferred_content_type = SupportedContentType[inferred_content_type.upper()].value.lower()
    elif inferred_content_type.lower() in SupportedImageType.list():
        inferred_content_type = "IMAGE"
    logging.debug("Inferred content type from file extension is %s", inferred_content_type)
    return inferred_content_type


def s3_format_uri(bucket_name, prefix=""):
    uri = "s3://" + bucket_name + "/" + prefix
    logging.debug("Formatted uri is %s", uri)
    return uri


def s3_get_basename(key):
    basename = PurePath(key).name
    return basename


def s3_parse_bucket_name_and_prefix(uri):
    if uri.startswith("s3a://"):
        uri = uri.replace("s3a://", "s3://")
    parse_result = urlparse(uri)
    bucket_name = parse_result.netloc
    # Replace only the first delimiter and not all as there could be path s3://bucket///folder
    prefix = parse_result.path.replace("/", "", 1)
    return bucket_name, prefix


def get_client_error_status_code(client_error):
    return client_error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", -1)


def create_and_upload_manifest(output_path: str, kms_key: str = None):
    """Collects the list of objects in output_path and writes a manifest file.

    The manifest format can be found under the `S3Uri` section in
    https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_S3DataSource.html
    """
    import boto3

    error_msg = "An error occurred when generating a manifest file. See the exception for more details: {error}"

    region_name = os.getenv("AWS_REGION")
    if region_name:
        s3_client = boto3.client("s3", region_name=region_name)
    else:
        s3_client = boto3.client("s3")

    bucket, prefix = s3_parse_bucket_name_and_prefix(output_path)
    prefix = prefix.rstrip("/") + "/"
    manifest_key = f"{prefix}manifest/data_wrangler_output.manifest"

    paginator = s3_client.get_paginator("list_objects")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    keys = []
    try:
        for page in page_iterator:
            for content in page["Contents"]:
                keys.append(content["Key"].removeprefix(prefix))
    except Exception as e:
        raise RuntimeError(error_msg.format(error=e))

    put_object_request = {
        "Body": json.dumps([{"prefix": f"s3://{bucket}/{prefix}"}] + keys),
        "Bucket": bucket,
        "Key": manifest_key,
    }
    if kms_key:
        put_object_request["SSEKMSKeyId"] = kms_key
        put_object_request["ServerSideEncryption"] = "aws:kms"
    try:
        s3_client.put_object(**put_object_request)
    except Exception as e:
        raise RuntimeError(error_msg.format(error=e))


def s3_replace_object_name(uri, object_name):
    bucket, prefix = s3_parse_bucket_name_and_prefix(uri)
    parsed_prefix = PurePath(prefix)
    prefix = str(parsed_prefix.with_name(object_name))
    return bucket, prefix


def s3_access_point_arn_match(uri: str) -> Optional[re.match]:
    pattern = re.compile(ACCESSPOINT_ARN_VERBOSE_REGEX, re.VERBOSE,)
    match = re.fullmatch(pattern, uri)  # fullmatch matches the whole string
    return match


def s3_access_point_arn_to_alias(arn: str, s3_control_client) -> str:
    """
    converts s3 acces_point arn (or uri) to alias. Eg:
    s3://<Accesspoint arn>/path/to/obj -> s3a://<accesspoint alias>/path/to/obj
    """
    match = s3_access_point_arn_match(arn)
    if match is None:
        raise OperatorCustomerError("The access point arn format is invalid.")

    account_id = match.group("account_id")
    access_point_name = match.group("name")
    object_path = match.group("path") if match.group("path") else ""

    response = s3_control_client.get_access_point(AccountId=account_id, Name=access_point_name)
    alias = response["Alias"]
    return os.path.join("s3://", alias, object_path)


def skip_lines_before_csv_header(spark, file_path, encoding, skip_lines):
    """Skips the number of rows before the header row: Reads the file as text. 
       The lambda function creates a new RDD of all the lines with index GE than the skip_lines arg.
       The returned RDD is then used as a path in the spark.read"""
    if isinstance(file_path, list):
        file_path = ",".join(file_path)  # sparkContext.binaryFiles cannot take a list, only comma separated string
    text_rdd = spark.sparkContext.binaryFiles(file_path).values().flatMap(lambda x: x.decode(encoding).splitlines())
    text_rdd = text_rdd.zipWithIndex().filter(lambda x: x[1] >= skip_lines).map(lambda x: x[0])

    return text_rdd


def split_s3_uri(s3_uri):
    parsed_uri = urlparse(s3_uri)
    # Extract bucket name and key from the S3 URI
    bucket_name = parsed_uri.netloc
    key = parsed_uri.path.lstrip("/")

    return bucket_name, key


import logging

import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, BinaryType, StructType

from pyspark.sql import DataFrame, types
from pyspark.ml.linalg import VectorUDT



def serialize_columns(df: DataFrame):
    """ Cast to string/json for serialization of vector/array/struct/binary type columns"""
    for field in df.schema.fields:
        if isinstance(field.dataType, StructType) or (
            isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType)
        ):
            df = df.withColumn(field.name, F.to_json(field.name))
        elif isinstance(field.dataType, VectorUDT) or isinstance(field.dataType, ArrayType):
            df = df.withColumn(field.name, F.col(field.name).cast(types.StringType()))
        elif isinstance(field.dataType, BinaryType):
            df = df.withColumn(field.name, F.base64(F.col(field.name)))

    return df


def coalesce_df(df: DataFrame, partitions=None):
    """ Coalesce df when the dataframe size is greater than threshold """
    curr_partitions = df.rdd.getNumPartitions()

    if partitions and isinstance(partitions, int):
        if partitions > 0:
            if curr_partitions < partitions:
                logging.info(
                    f"Requested partitions ({partitions}) is greater than the current number of partitions ({curr_partitions}). Repartitioning"
                )
                return df.repartition(partitions)
            if curr_partitions > partitions:
                return df.coalesce(partitions)
            return df
        else:
            logging.info(f"Requested partitions must be greater than 0. Ignoring: {partitions}")

    if curr_partitions != 1:
        estimated_df_size_bytes = estimate_dataframe_size_bytes(df)

        # perform coalesce when the dataset is below the threshold
        if estimated_df_size_bytes <= DATAFRAME_AUTO_COALESCING_SIZE_THRESHOLD:
            logging.info(
                "Performing auto coalescing to 1 partition for data with estimated size "
                + f"below {DATAFRAME_AUTO_COALESCING_SIZE_THRESHOLD} bytes"
            )
            return df.coalesce(1)
    return df



# 1500 non-null records provides high confidence (>99.99%) for type inference
# Ratio of 0.3 is used when fewer than 5000 records are available
DEFAULT_TYPE_INFERENCE_SAMPLE_SIZE = 5000
NON_NULL_RATIO = 0.3

# infer and cast operator thresholds
NUMERIC_THRESHOLD = 0.8
INTEGER_THRESHOLD = 0.8
BOOL_THRESHOLD = 0.8
DATE_THRESHOLD = 0.8
DATETIME_THRESHOLD = 0.8

# logical type inference thresholds
LOGICAL_TYPE_INFERENCE_SAMPLE_SIZE = 1500000
CATEGORICAL_PERCENTAGE_THRESHOLD = 0.10
LOGICAL_NUMERIC_THRESHOLD = 0.9
CATEGORICAL_THRESHOLD = 100


def type_inference(df):  # noqa: C901 # pylint: disable=R0912
    """Core type inference logic

    Args:
        df: spark dataframe

    Returns: dict a schema that maps from column name to mohave datatype

    """
    columns_to_infer = [escape_column_name(col) for (col, col_type) in df.dtypes if col_type == "string"]
    pandas_df = df[columns_to_infer].toPandas()
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(get_proposed_type, series) for _, series in pandas_df.iteritems()]
        result_types = [f.result() for f in futures]
    column_names_types_map = {column: col_type for column, col_type in zip(pandas_df.columns, result_types)}

    for f in df.schema.fields:
        if f.name not in column_names_types_map:
            if isinstance(f.dataType, IntegralType):
                column_names_types_map[f.name] = MohaveDataType.LONG.value
            elif isinstance(f.dataType, FractionalType):
                column_names_types_map[f.name] = MohaveDataType.FLOAT.value
            elif isinstance(f.dataType, StringType):
                column_names_types_map[f.name] = MohaveDataType.STRING.value
            elif isinstance(f.dataType, BooleanType):
                column_names_types_map[f.name] = MohaveDataType.BOOL.value
            elif isinstance(f.dataType, TimestampType):
                column_names_types_map[f.name] = MohaveDataType.DATETIME.value
            elif isinstance(f.dataType, ArrayType):
                column_names_types_map[f.name] = MohaveDataType.ARRAY.value
            elif isinstance(f.dataType, StructType):
                column_names_types_map[f.name] = MohaveDataType.STRUCT.value
            else:
                # unsupported types in mohave
                column_names_types_map[f.name] = MohaveDataType.OBJECT.value

    return column_names_types_map


def _is_numeric_single(x):
    try:
        if isinstance(x, str) and "_" in x:
            return False
        x_float = float(x)
        return np.isfinite(x_float)
    except ValueError:
        return False
    except TypeError:  # if x = None
        return False


def count_numeric(x):
    """count number of numeric element

    Args:
        x: numpy array

    Returns: int

    """
    castables = np.vectorize(_is_numeric_single, otypes=[bool])(x)
    return np.count_nonzero(castables)


def _is_integer_single(x):
    try:
        if not _is_numeric_single(x):
            return False
        return float(x) == int(x)
    except ValueError:
        return False
    # except TypeError:  # if x = None
    #     return False


def count_integer(x):
    castables = np.vectorize(_is_integer_single, otypes=[bool])(x)
    return np.count_nonzero(castables)


def _is_boolean_single(x):
    boolean_list = ["true", "false"]
    try:
        is_boolean = x.lower() in boolean_list
        return is_boolean
    except ValueError:
        return False
    except TypeError:  # if x = None
        return False
    except AttributeError:
        return False


def count_boolean(x):
    castables = np.vectorize(_is_boolean_single, otypes=[bool])(x)
    return np.count_nonzero(castables)


def count_null_like(x):  # noqa: C901
    def _is_empty_single(x):
        try:
            return bool(len(x) == 0)
        except TypeError:
            return False

    def _is_null_like_single(x):
        try:
            return bool(null_like_regex.match(x))
        except TypeError:
            return False

    def _is_whitespace_like_single(x):
        try:
            return bool(whitespace_regex.match(x))
        except TypeError:
            return False

    null_like_regex = re.compile(r"(?i)(null|none|nil|na|nan)")  # (?i) = case insensitive
    whitespace_regex = re.compile(r"^\s+$")  # only whitespace

    empty_checker = np.vectorize(_is_empty_single, otypes=[bool])(x)
    num_is_null_like = np.count_nonzero(empty_checker)

    null_like_checker = np.vectorize(_is_null_like_single, otypes=[bool])(x)
    num_is_null_like += np.count_nonzero(null_like_checker)

    whitespace_checker = np.vectorize(_is_whitespace_like_single, otypes=[bool])(x)
    num_is_null_like += np.count_nonzero(whitespace_checker)
    return num_is_null_like


def count_null(x):
    with pd.option_context("mode.use_inf_as_na", True):
        return np.count_nonzero(pd.isnull(x))


def _is_date_single(x):
    try:
        return bool(date.fromisoformat(x))  # YYYY-MM-DD
    except ValueError:
        return False
    except TypeError:
        return False


def count_date(x):
    return np.count_nonzero(np.vectorize(_is_date_single, otypes=[bool])(x))


def count_datetime(x):
    # detects all possible convertible datetimes, including multiple different formats in the same column
    return pd.to_datetime(x, cache=True, errors="coerce").notnull().sum()


def is_insufficient_records(
    count_not_null, count_string, sample_size=DEFAULT_TYPE_INFERENCE_SAMPLE_SIZE,
):
    if count_not_null == 0:
        return True
    min_non_null_count = sample_size * NON_NULL_RATIO
    is_less_than_min_count = count_string >= sample_size and count_not_null < min_non_null_count
    is_less_than_ratio = count_string < sample_size and (count_not_null / count_string) < NON_NULL_RATIO
    return is_less_than_min_count or is_less_than_ratio


def get_col_insights(column: np.ndarray):
    return {
        "count_string": len(column),
        "count_numeric": count_numeric(column),
        "count_integer": count_integer(column),
        "count_boolean": count_boolean(column),
        "count_null_like": count_null_like(column),
        "count_null": count_null(column),
    }


def get_proposed_type(column: np.ndarray):
    """Perform type inference based on column insights."""
    insights = get_col_insights(column)
    proposed = MohaveDataType.STRING.value

    count_not_null = insights["count_string"] - (insights["count_null"] + insights["count_null_like"])

    if is_insufficient_records(count_not_null, insights["count_string"]):
        # if we don't have sufficient records to make a confident type inference, default to string
        return MohaveDataType.STRING.value

    if (insights["count_numeric"] / count_not_null) > NUMERIC_THRESHOLD:
        if (insights["count_integer"] / insights["count_numeric"]) > INTEGER_THRESHOLD:
            return MohaveDataType.LONG.value
        return MohaveDataType.FLOAT.value

    if (insights["count_boolean"] / count_not_null) > BOOL_THRESHOLD:
        return MohaveDataType.BOOL.value

    # compute count_datetime and count_date only if none of the above proposed types
    insights["count_date"], insights["count_datetime"] = count_date(column), count_datetime(column)
    if (insights["count_date"] / count_not_null) > DATE_THRESHOLD:
        # datetime - date is # of rows with time info
        # if even one value w/ time info in a column with mostly dates, choose datetime
        if (insights["count_datetime"] - insights["count_date"]) > 0:
            return MohaveDataType.DATETIME.value
        return MohaveDataType.DATE.value
    if (insights["count_datetime"] / count_not_null) > DATETIME_THRESHOLD:
        return MohaveDataType.DATETIME.value
    return proposed


def cast_df(df, schema, spark):
    """Cast dataframe from given schema

    Args:
        df: spark dataframe
        schema: schema to cast to. It map from df's col_name to mohave datatype
        spark: spark session

    Returns: casted dataframe

    """
    # col name to spark data type mapping
    col_to_spark_data_type_map = {}

    # get spark dataframe's actual datatype
    fields = df.schema.fields
    for f in fields:
        col_to_spark_data_type_map[f.name] = f.dataType
    cast_expr = []

    # iterate given schema and cast spark dataframe datatype
    for col_name in schema:
        mohave_data_type_from_schema = MohaveDataType(schema.get(col_name, MohaveDataType.OBJECT.value))
        if mohave_data_type_from_schema == MohaveDataType.DATETIME:
            df = cast_datetime_column(df, col_name, spark)
            expr = f"`{col_name}`"  # keep the column in the SQL query that is run below
        elif mohave_data_type_from_schema != MohaveDataType.OBJECT:
            spark_data_type_from_schema = MOHAVE_TO_SPARK_TYPE_MAPPING.get(mohave_data_type_from_schema)
            if not spark_data_type_from_schema:
                raise KeyError(f"Key {mohave_data_type_from_schema} not present in MOHAVE_TO_SPARK_TYPE_MAPPING")
            # Only cast column when the data type in schema doesn't match the actual data type
            # and data type is not Array or Struct
            if spark_data_type_from_schema not in [ArrayType, StructType] and not isinstance(
                col_to_spark_data_type_map[col_name], spark_data_type_from_schema
            ):
                # use spark-sql expression instead of spark.withColumn to improve performance
                expr = f"CAST (`{col_name}` as {SPARK_TYPE_MAPPING_TO_SQL_TYPE[spark_data_type_from_schema]})"
            else:
                # include column that has same dataType as it is
                expr = f"`{col_name}`"
        else:
            # include column that has same mohave object dataType as it is
            expr = f"`{col_name}`"
        cast_expr.append(expr)
    if len(cast_expr) != 0:
        df = df.selectExpr(*cast_expr)
    return df, schema


def validate_schema(df, schema):
    """Validate if every column is covered in the schema

    Args:
        schema ():
    """
    columns_in_df = df.columns
    columns_in_schema = schema.keys()

    if len(columns_in_df) != len(columns_in_schema):
        raise ValueError(
            f"Invalid schema column size. "
            f"Number of columns in schema should be equal as number of columns in dataframe. "
            f"schema columns size: {len(columns_in_schema)}, dataframe column size: {len(columns_in_df)}"
        )

    for col in columns_in_schema:
        if col not in columns_in_df:
            raise ValueError(
                f"Invalid column name in schema. "
                f"Column in schema does not exist in dataframe. "
                f"Non-existed columns: {col}"
            )


def infer_logical_types(df, check_null_records: bool = True):
    n_rows, n_cols = df.count(), len(df.columns)
    # When there are no more than 1000 columns, we use no more than 1.5MM
    # operations. If there are more than 1000 columns we use no fewer than 1500
    # rows unless there are fewer than 1500 rows in the dataframe, in which case
    # we use them all. The minimum of 1500 was chosen to balance accuracy and
    # performance.
    sample_size = min(n_rows, max(LOGICAL_TYPE_INFERENCE_SAMPLE_SIZE // n_cols, 1500))
    df = df.limit(sample_size)
    df_pd = df.toPandas()
    numeric_cols, categorical_cols, string_cols, cols_to_infer = [], [], [], []

    for col, _ in df.dtypes:
        # Filter out unsupported types
        if not isinstance(df.schema[col].dataType, tuple(SUPPORTED_TYPES)):
            continue
        if isinstance(df.schema[col].dataType, BooleanType):
            categorical_cols.append(col)
            continue

        column = df_pd[col].values
        count_total = len(column)
        count_not_null = count_total - (count_null(column) + count_null_like(column))

        if check_null_records and is_insufficient_records(
            count_not_null, count_total, sample_size=LOGICAL_TYPE_INFERENCE_SAMPLE_SIZE
        ):
            # if we don't have sufficient records to make a confident type inference, default to string when check_null is True
            string_cols.append(col)
            continue

        if count_not_null == 0:
            string_cols.append(col)
            continue

        if count_numeric(column) / count_not_null > LOGICAL_NUMERIC_THRESHOLD:
            numeric_cols.append(col)
            continue

        number_of_categories = df_pd[col].nunique()
        if (
            number_of_categories > CATEGORICAL_THRESHOLD
            or number_of_categories / count_not_null > CATEGORICAL_PERCENTAGE_THRESHOLD
        ):
            string_cols.append(col)
            continue
        else:
            categorical_cols.append(col)
            continue
    return numeric_cols, categorical_cols, string_cols


def s3_source(spark, mode, dataset_definition, flow_parameters=None):
    """Represents a source that handles sampling, etc."""


    s3_data_type = dataset_definition["s3ExecutionContext"].get("s3DataType", "S3Prefix")


    if s3_data_type == "S3Prefix":
        path = dataset_definition["s3ExecutionContext"]["s3Uri"].replace("s3://", "s3a://")


    content_type = dataset_definition["s3ExecutionContext"]["s3ContentType"].upper()
    recursive = "true" if dataset_definition["s3ExecutionContext"].get("s3DirIncludesNested") else "false"
    adds_filename_column = dataset_definition["s3ExecutionContext"].get("s3AddsFilenameColumn", False)
    role_arn = dataset_definition["s3ExecutionContext"].get("s3RoleArn", None)
    # check to make sure role arn is not empty string or any other falsy value
    role_arn = role_arn if role_arn else None
    try:
            if content_type == SupportedContentType.CSV.value:
                has_header = dataset_definition["s3ExecutionContext"]["s3HasHeader"]
                field_delimiter = dataset_definition["s3ExecutionContext"].get("s3FieldDelimiter", ",")
                csv_encoding_type = dataset_definition["s3ExecutionContext"].get("s3CsvEncodingType", "utf-8")
                csv_encoding_type = (
                    csv_encoding_type.replace("_", "-").lower()
                    if isinstance(csv_encoding_type, str)
                    else csv_encoding_type
                )
                skip_lines = dataset_definition["s3ExecutionContext"].get("s3SkipLines", 0)
                skip_lines = skip_lines if isinstance(skip_lines, int) else 0  # skip lines must be int
                multi_line = dataset_definition["s3ExecutionContext"].get("s3MultiLine", False)
                if not field_delimiter:
                    field_delimiter = ","
                if skip_lines > 0:
                    path = skip_lines_before_csv_header(spark, path, csv_encoding_type, skip_lines)
                df = spark.read.option("recursiveFileLookup", recursive).csv(
                    path=path,
                    header=has_header,
                    escape='"',
                    quote='"',
                    sep=field_delimiter,
                    mode="PERMISSIVE",
                    encoding=csv_encoding_type,
                    multiLine=multi_line,
                )
            elif content_type == SupportedContentType.PARQUET.value:
                if isinstance(path, list):
                    # For Parquet only, Unpack the list because Spark read does not take a list as input
                    df = spark.read.option("recursiveFileLookup", recursive).parquet(*path)
                else:
                    df = spark.read.option("recursiveFileLookup", recursive).parquet(path)
            elif content_type == SupportedContentType.JSON.value:
                df = spark.read.option("multiline", "true").option("recursiveFileLookup", recursive).json(path)
            elif content_type == SupportedContentType.JSONL.value:
                df = spark.read.option("multiline", "false").option("recursiveFileLookup", recursive).json(path)
            elif content_type == SupportedContentType.ORC.value:
                df = spark.read.option("recursiveFileLookup", recursive).orc(path)


            if adds_filename_column:
                df = add_filename_column(df)
            return default_spark(df)
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading files from S3") from e


def infer_and_cast_type(
    df, spark, inference_data_sample_size=DEFAULT_TYPE_INFERENCE_SAMPLE_SIZE, trained_parameters=None, **kwargs,
):
    """Infer column types for spark dataframe and cast to inferred data type.

    Args:
        df: spark dataframe
        spark: spark session
        inference_data_sample_size: number of row data used for type inference
        trained_parameters: trained_parameters to determine if we need infer data types

    Returns: a dict of pyspark df with column data type casted and trained parameters

    """

    # if trained_parameters is none or doesn't contain schema key, then type inference is needed
    if trained_parameters is None or not trained_parameters.get("schema", None):
        # limit first 1000 rows to do type inference
        limit_df = df.limit(inference_data_sample_size)
        schema = type_inference(limit_df)
    else:
        schema = trained_parameters["schema"]
        try:
            validate_schema(df, schema)
        except ValueError as e:
            raise OperatorCustomerError(e)
    try:
        df, schema = cast_df(df, schema, spark)
    except (AnalysisException, ValueError) as e:
        raise OperatorCustomerError(e)
    trained_parameters = {"schema": schema}
    return default_spark_with_trained_parameters(df, trained_parameters)


def manage_columns(df, spark, **kwargs):

    return dispatch(
        "operator",
        [df],
        kwargs,
        {
            "Drop column": (manage_columns_drop_column, "drop_column_parameters"),
            "Duplicate column": (manage_columns_duplicate_column, "duplicate_column_parameters"),
            "Rename column": (manage_columns_rename_column, "rename_column_parameters"),
            "Move column": (manage_columns_move_column, "move_column_parameters"),
            "Concatenate columns": (manage_columns_concat_columns, "concat_columns_parameters"),
        },
    )


def manage_rows(df, spark, **kwargs):

    return dispatch(
        "operator",
        [df],
        kwargs,
        {
            "Shuffle": (manage_rows_shuffle, "shuffle_parameters"),
            "Sort": (manage_rows_sort, "sort_parameters"),
            "Drop duplicates": (manage_rows_drop_duplicates, "drop_duplicates_parameters"),
            "Drop empty": (manage_rows_drop_empty, "drop_empty_parameters"),
            "Filter": (filter_by_value, "filter_parameters"),
        },
    )


def handle_missing(df, spark, **kwargs):

    # Handle the old interface for Drop missing by converting to new interface
    if kwargs["operator"] == "Drop missing":
        drop_missing_params = kwargs.get("drop_missing_parameters")
        drop_rows_params = drop_missing_params.get("drop_rows_parameters") if drop_missing_params else None
        if drop_rows_params and drop_missing_params.get("dimension") == "Drop Rows":
            kwargs["drop_missing_parameters"] = drop_rows_params

    return dispatch(
        "operator",
        [df],
        kwargs,
        {
            "Impute": (handle_missing_impute, "impute_parameters"),
            "Fill missing": (handle_missing_fill_missing, "fill_missing_parameters"),
            "Add indicator for missing": (
                handle_missing_add_indicator_for_missing,
                "add_indicator_for_missing_parameters",
            ),
            "Drop missing": (handle_missing_drop_rows, "drop_missing_parameters"),
        },
    )


def encode_categorical(df, spark, **kwargs):

    return dispatch(
        "operator",
        [df],
        kwargs,
        {
            "Ordinal encode": (encode_categorical_ordinal_encode, "ordinal_encode_parameters"),
            "One-hot encode": (encode_categorical_one_hot_encode, "one_hot_encode_parameters"),
            "Similarity encode": (encode_categorical_similarity_encode, "similarity_encode_parameters"),
        },
    )


def handle_outliers(df, spark, **kwargs):

    return dispatch(
        "operator",
        [df],
        kwargs,
        {
            "Standard deviation numeric outliers": (
                handle_outliers_standard_deviation_numeric_outliers,
                "standard_deviation_numeric_outliers_parameters",
            ),
            "Robust standard deviation numeric outliers": (
                handle_outliers_robust_standard_deviation_numeric_outliers,
                "robust_standard_deviation_numeric_outliers_parameters",
            ),
            "Quantile numeric outliers": (
                handle_outliers_quantile_numeric_outliers,
                "quantile_numeric_outliers_parameters",
            ),
            "Min-max numeric outliers": (
                handle_outliers_min_max_numeric_outliers,
                "min_max_numeric_outliers_parameters",
            ),
            "Replace rare values": (handle_outliers_replace_rare_values, "replace_rare_values_parameters"),
        },
    )


def s3_destination(input, spark, mode, output_config: dict):
    """S3 destination operator for writing df or visualization output to S3 as part of graph evaluation
    Args:
        input: df or collections.Mapping, input to write to s3 to
        output_config: dict, contains output path in s3 etc
    """

    is_network_iso_mode = False

    if s3_access_point_arn_match(output_config["output_path"]) and not is_network_iso_mode:
        s3_control_client = boto3.client("s3control", region_name=os.getenv("AWS_REGION"))
        output_config["output_path"] = s3_access_point_arn_to_alias(output_config["output_path"], s3_control_client)

    if isinstance(input, DataFrame):
        return write_df_to_s3(input, output_config)
    elif isinstance(input, collections.Mapping):
        return write_visualizations_to_s3(input, output_config, spark)
    else:
        raise OperatorCustomerError(f"{type(input)} not supported as destination.")


op_1_output = s3_source(spark=spark, mode=mode, **{'dataset_definition': {'__typename': 'S3CreateDatasetDefinitionOutput', 'datasetSourceType': 'S3', 'name': 'hotel_bookings.csv', 'description': None, 's3ExecutionContext': {'__typename': 'S3ExecutionContext', 's3Uri': 's3://datawrangler-hotelbooking-ml-deep-pat/hotel_bookings.csv', 's3ContentType': 'csv', 's3HasHeader': True, 's3FieldDelimiter': ',', 's3CsvEncodingType': 'utf-8', 's3DirIncludesNested': False, 's3AddsFilenameColumn': False, 's3RoleArn': '', 's3SkipLines': 0}}})
op_2_output = infer_and_cast_type(op_1_output['default'], spark=spark, **{})
op_10_output = manage_columns(op_2_output['default'], spark=spark, **{'operator': 'Drop column', 'drop_column_parameters': {'column_to_drop': ['babies', 'hotel', 'arrival_date_day_of_month', 'meal', 'previous_cancellations', 'arrival_date_month', 'days_in_waiting_list', 'arrival_date_week_number', 'arrival_date_year']}})
op_11_output = manage_rows(op_10_output['default'], spark=spark, **{'operator': 'Drop duplicates', 'drop_duplicates_parameters': {}, 'sort_parameters': {}})
op_12_output = handle_missing(op_11_output['default'], spark=spark, **{'operator': 'Impute', 'impute_parameters': {'column_type': 'Numeric', 'numeric_parameters': {'input_column': ['agent'], 'strategy': 'Approximate Median'}}})
op_13_output = encode_categorical(op_12_output['default'], spark=spark, **{'operator': 'One-hot encode', 'one_hot_encode_parameters': {'input_column': ['market_segment', 'assigned_room_type', 'deposit_type', 'customer_type', 'is_repeated_guest'], 'invalid_handling_strategy': 'Keep', 'drop_last': False, 'output_style': 'Columns'}, 'ordinal_encode_parameters': {'invalid_handling_strategy': 'Replace with NaN'}})
op_14_output = manage_columns(op_13_output['default'], spark=spark, **{'operator': 'Rename column', 'rename_column_parameters': {'columns_to_rename': [{'input_column': 'market_segment_Offline TA/TO', 'new_name': 'market_segment_Offline_TA_TO'}, {'input_column': 'deposit_type_No Deposit', 'new_name': 'deposit_type_No_Deposit'}, {'input_column': 'deposit_type_Non Refund', 'new_name': 'deposit_type_Non_Refund'}], 'input_column': 'market_segment_Online TA', 'new_name': 'market_segment_Online_TA'}, 'drop_column_parameters': {}})
op_15_output = handle_outliers(op_14_output['default'], spark=spark, **{'operator': 'Standard deviation numeric outliers', 'standard_deviation_numeric_outliers_parameters': {'input_column': ['children'], 'standard_deviations': 4, 'fix_method': 'Remove'}})
op_16_output = s3_destination(op_15_output['default'], spark=spark, mode=mode, **{'output_config': {'compression': 'none', 'output_path': 's3://datawrangler-hotelbooking-ml-deep-pat/', 'output_content_type': 'CSV', 'delimiter': ','}})

#  Glossary: variable name to node_id
#
#  op_1_output: f1dd827e-73a9-4d00-9f6a-8186cf0f75c3
#  op_2_output: 8ff66a2c-594b-4612-957c-aca9f722fbc7
#  op_10_output: c2e8e92a-e8f2-4689-bee1-53993fbadd6b
#  op_11_output: 25372b47-bec9-4d4c-aa5f-be17195c87f9
#  op_12_output: d6f2a0c4-c79f-4fe4-9b3e-c150b7da12ed
#  op_13_output: b9568200-b47b-46ec-b9f3-a88ff16960fd
#  op_14_output: 7374c62a-245a-4e10-bea1-2a3e2ac76ff1
#  op_15_output: 2aa6b317-4d04-4d07-9631-6b73bc062136
#  op_16_output: 5a4946b4-de47-46bb-a8f0-22bb46f4f901