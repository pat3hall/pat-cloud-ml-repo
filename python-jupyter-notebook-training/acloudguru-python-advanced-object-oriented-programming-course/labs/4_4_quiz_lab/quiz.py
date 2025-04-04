from random import sample

class Quiz:
    def __init__(self, questions, passing_percent):
        self.questions = questions
        self.passing_percent = passing_percent
        self.actual_score = 0

    def start(self):
        for index, question in enumerate(list(sample(self.questions, len(self.questions)))):
            print (f"{index + 1}. {question}")
            answer = input("Answer: ")
            # with open ("answer.txt", "a") as f:
            #    f.write (answer + "\n")
            question.select(answer)
            print("\n\n")

    def score(self):
        self.actual_score = 0
        for question in self.questions:
            if question.grade():
                self.actual_score += 1

        return self.actual_score

    def grade (self):
        self.score()
        percent = round(self.actual_score / len(self.questions), 2)
        return (percent, percent >= self.passing_percent)


from random import sample

class Question:
    def __init__(self, question_text, choices, correct_answer) -> None:
        self.question_text = question_text
        #self.choices = [str(element) for element in choices]
        self.choices = choices
        self.correct_answer = str(correct_answer)
        self.selected_answer = None

    def __str__(self) -> str:
        output = self.question_text + "\n\n"
        choices = self.choices
        # index through randomize choices list
        for index, choice in enumerate(choices, start=0 ):
            # index + 1 because index starts at zero
            output += f"{index + 1}. {choice}\n"
        return output

    def select(self, selected_answer) -> None:
        self.selected_answer = str(selected_answer)

    def grade(self) -> bool:
        return self.correct_answer == self.selected_answer

class TrueFalseQuestion(Question):
    def __init__(self, question_text, correct_answer) -> None:
        question_text = self.__prefix_if_necessary(question_text)
        super().__init__(question_text=question_text, choices = [True, False], correct_answer = correct_answer)
        #self.choices.sort()

    def __prefix_if_necessary(self, question_text) -> str:
        if question_text.lower().startswith("false") or question_text.lower().startswith("true"):
            return question_text
        else:
            return f"True/False: {question_text}"


class MultipleSelectQuestion(Question):
    def __init__(self, question_text, choices, correct_answers) -> None:
        question_text = self.__add_text(question_text, correct_answers)
        super().__init__(question_text=question_text, choices=choices, correct_answer=correct_answers.sort())
        self.correct_answer = self.__sort_string_list(correct_answers)

    def __add_text(self, question_text, correct_answer):
        return f"{question_text} (select {len(correct_answer)})"

    def __sort_string_list (self, list_of_items):
         return list(sorted(map(str, list_of_items)))

    def select(self, selected_answer) -> None:
        self.selected_answer = self.__sort_string_list(selected_answer)

if __name__ == "__main__":
    #from questions import Question, TrueFalseQuestion
    question1 = Question("What's the answer to life, the universe, and everything?", [42, "Silver", "Wood", True], 42)
    question2 = TrueFalseQuestion("Ice cream is the best dessert.", True)
    quiz = Quiz(questions=[question1, question2], passing_percent=0.50)
    
    from essay_question import EssayQuestion
    essay = EssayQuestion("How would you go about building a web application?")
    quiz.questions.append(essay)
    print("\nStarting quiz: answers: 42, True, 'My answer is awesome', Y\n")
    quiz.start()
    print(f"Checking quiz score: {quiz.score()}")
    print(f"Checking quiz grade: {quiz.grade()}")



    
    
     