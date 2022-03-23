from hstest import *
import re

dict_answers_CORRECT = {
    "samplesize": 400,
    "controlgroup": 400,
    "experimentalgroup": 400
}


class SampleSize(StageTest):
    @dynamic_test
    def test(self):
        pr = TestedProgram()
        output = pr.start().replace(" ", "").lower()

        if len(output.rstrip()) == 0:
            raise WrongAnswer("Seems like your program does not show any output.")

        output_splitted = output.split("\n")
        # let's remove all empty lines from user's output
        output_splitted = [line for line in output_splitted if line.strip() != ""]

        if len(output_splitted) != 3:
            raise WrongAnswer(f"Wrong number of non-empty lines in the output of you program.\n"
                              f"Expected 3, found {len(output_splitted)}.\n"
                              f"Check the output format in the Examples section.")
        for key_word in ["Sample size", "Control group", "Experimental group"]:
            key_word_raw = key_word.replace(" ", "").lower()
            # check that a key word occurs only once
            if output.count(key_word_raw) != 1:
                raise WrongAnswer(f"Substring \"{key_word}\" should occur once in the output of your program.\n"
                                  f"Found {output.count(key_word_raw)} occurrences.")
            # check the value related to the key word (size of a group)
            answer_user = re.search(f'{key_word_raw}:([+-]?(?:[0-9]*[.])?[0-9]+)', output)
            if answer_user is None:
                raise WrongAnswer(f"Didn't find the value for {key_word.lower()}.\n"
                                  f"Note that you should present its value in the following format:\n"
                                  f"    \"{key_word}: <calculated value>\"")
            if float(answer_user.group(1)) != dict_answers_CORRECT[key_word_raw]:
                raise WrongAnswer(f"The value for {key_word.lower()} is wrong.")

        else:
            return CheckResult.correct()


if __name__ == '__main__':
    SampleSize().run_tests()
