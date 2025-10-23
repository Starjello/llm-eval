The correct_prompts and correct_generated_solutions contain only successful attempts. The normal prompts and generated_solutions is exclusively incorrect attempts.
My python environment contains 4 folders for said prompts and generated solution storage, each have their own subfolders "gemini" and "openai" for sorting
The main code is runEval.py, the flush.py file empties the storage files. No other file is worth looking at

In order to run the code you need:
To use the set command in terminal to set a value to the api keys
Have installed google gemini and openai packages
You may need to create the same folder structure I described earlier including a combo.passk_results.json file to catch the summary report but I am not entirely sure


Packages needed: google-genai, openai, humaneval
