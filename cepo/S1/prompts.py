def get_numeric_type_prompt(parsed_answer):
    """Determines if the problem type is numeric. If not then it is classified as Equation"""
    numeric_type_prompt = f"""
        [System Instruction]: You are an AI assistant that follows user instructions carefully. Format responses correctly and adhere to all constraints.
        I want you to follow the following format while outputing the label:

        [Task]: You are a text classifier which takes an piece of text as input, and returns whether the text is numeric type or equation type. I will define the two labels allowed in the output:
        1. numeric: It is a piece of text which only contains numbers(int , float etc.). Even matrices are considered to be numeric types. Now, there can be any operations in the text , where each operation is between numeric types or expressions which can be simplified to numeric types. Those operations can be written using latex tags as welll. Simply, any text which can be simplified to a number is numeric type.
        2. equation: Any piece of text which is an expresission of variables, involving operations, numbers etc. can be called equation type. Or simply an expression/text which you know cant be simplified down to a numeric type.

        I want you to reason about how you found out the given text belongs to which class as well. You should logically think and maybe correct yourself too, if you think you reasoning is wrong.

        All the reasoning steps should be enclosed between <reasoning> and </reasoning> tags

        Finally your classification answer should be between <label> and </label>.

        [Examples]:

        Example 1:
        Input text: 10\\sqrt{2}
        Output: <reasoning>
        Since the text can be reduced into a number which is 14.14, thus it is of numeric type.
        </reasoning>
        <label>numeric</label>

        Example 2:
        Input text: 150
        Output: <reasoning>
        The input text contains only a number, so it is numeric type.
        </reasoning>
        <label>numeric</label>

        Example 3:
        Input text: p - \\frac{2}{3}
        Output:<reasoning>
        Although there is a number in there i.e. \\frac{2}{3}, but the variable p makes it an equation.
        </reasoning>
        <label>equation</label>

        Notes: 
        - i will refer to the imaginery number, it is not a variable

        Classify the following text: {parsed_answer}

        [End of Instruction]

        """
    return numeric_type_prompt 

def get_compare_numeric_answers_prompt(answer):
    compare_numeric_answers_prompt = f"""
        Take the following mathematical expression: {answer} and convert it to the python code 
        that would evaluate to the equivalent value.
        If we run your output it be equivalent to the mathematical expression as the mathematical expression.
        We are converting to python code so that we can do exact comparisons between two values without 
        relying on comparing strings. 
        Do NOT include ANY additional code except the final answer (and any imports required).
        Use standard libraries (such as math) and numpy before you use any other packages.
        Store the final answer in a variable `final_answer`
    """
    return compare_numeric_answers_prompt 

def get_compare_equation_answers_prompt(answer, expected_answer):
    compare_equation_answers_prompt = f"""
        Compare the following mathematical expressions, and determine if they are equivalent. 
        Note that they might use slightly different syntax or have different number of spaces 
        or different variable names, but if they are equivalent mathematically, then they should be considered equivalent.
        - Answer: {answer}
        - Expected Answer: {expected_answer}
        Explain your reasoning as to why you think they are equivalent or not inside 
        of <reasoning> and </reasoning> tags. Then provide your final answer inside of <label> and </label> tags. 
        The final answer should be Yes or No.
    """
    return compare_equation_answers_prompt