import re

class ModelClass(type):
    def __new__(cls, name, based, dct):
        final = super().__new__(cls, name, based, dct)

        defaults = {
            key: value
            for key, value in final.__dict__.items()
            if not key.startswith("_")
        }

        init = final.__create_init(final.__annotations__, defaults)

        if not final.__dict__.get("__tablename__"):
            # print(f"final.__snake_case(final.__name__): {final.__snake_case(final.__name__)}")
            setattr(final, "__tablename__", final.__snake_case(final.__name__))

        tablename = final.__dict__["__tablename__"]
        
        setattr(final, "__init__", init)

        for field_name, field_type, in final.__annotations__.items():
            (func_name, func) = final.__create_get_by(tablename, field_name, field_type)
            setattr(final, func_name, func)

        return final
    
    @staticmethod
    def __create_init(annotations, defaults, return_type=None):
        name = "__init__"
        args =["self"]
        default_args = []
        body_lines = []

        for key, value in annotations.items():
            if key in defaults.keys():
                default_args.append(f"{key}:{value.__name__}={repr(defaults[key])}")
            else:
                args.append(f"{key}:{value.__name__}")
            
            body_lines.append(f"self.{key} = {key}")

        args = ", ".join(args + default_args)
        body = "\n ".join(body_lines)

        text = f"def {name}({args})->{return_type}:\n {body}"
        # run the code to create the method
        exec(text)
        return locals()[name]
    
    @staticmethod
    def __snake_case(value):
        # change from "OtherExample" -> _other_example -> other_example
         return re.sub("([A-Z])", r"_\1", value).lower().strip("_")
    
    @staticmethod
    def __create_get_by(tablename, field_name, field_type):
        base_query = f"select * from {tablename} where {field_name} = "
        body = None

        name = f"get_by_{field_name}"

        if field_type == str:
            body = f'value = value.replace("\'", "\'\'")\n return f"{base_query}\'{{value}}\'"'
        elif field_type == bool:
            body = f"value = 'true' if value else 'false'\n return f\"{base_query}{{value}}\""
        else:
            body = f' return f"{base_query}\'{{value}}\'"'

        text = f"def {name}(value):\n {body}"
        # print(f"\ntext:\n{text}\n")
        exec(text)
        # print (f"\nname: {name}, locals()[name]: {locals()[name]}\n")
        return (name, locals()[name])


if __name__ == '__main__':
    class Post(metaclass=ModelClass):
        __tablename__ = "posts"
        title: str
        content: str
        published: bool

    class OtherExample(metaclass=ModelClass):
        title: str
        content: str
        published: bool
    
    post = Post("New Title", "This is my content", False)
    print(f"post.__tablename__: {post.__tablename__} \t\t\texpected value: 'posts'")
    print(f"post.title:         {post.title} \t\t\texpected value: 'New Title'")
    print(f"post.content:       {post.content} \t\texpected value: 'This is my content'")
    print(f"post.published:     {post.published} \t\t\texpected value: 'False'")

    print(f"OtherExample.__tablename__: {OtherExample.__tablename__} \texpected value: 'other_example'")


    print("\ninfo = dir(Post)")
    info = dir(Post)
    print(f'"get_by_title" in info: {"get_by_title" in info} \t\t\texpected value: "True"')
    print(f'"get_by_content" in info: {"get_by_content" in info} \t\t\texpected value: "True"')
    print(f'"get_by_published" in info: {"get_by_published" in info} \t\texpected value: "True"')
    print(f'\nPost.get_by_title("My Title")         {Post.get_by_title("My Title")}')
    print('                     expected value:  select * from posts where title = \'My Title\'')
    print(f'\nPost.get_by_content("Some content")   {Post.get_by_content("Some content")}')
    print('                     expected value:  select * from posts where content = \'Some content\'')
    print(f'\nPost.get_by_published(False)          {Post.get_by_published(False)}')
    print('                     expected value:  select * from posts where published = false')


