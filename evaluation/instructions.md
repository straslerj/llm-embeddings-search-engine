To use the evaluation scripts:

1. Run the Makefile
   1. Make sure you're in `evaluation/`
   2. Create a set of questions in `questions/` as a CSV containing question,answer.
   3. Run the Makefile: `make MODE=<mode>` where `<mode>` is either `syllabus` or `nfl`
2. This will run the create_responses script and evaluation script automatically.
   1. create_responses writes a CSV to `responses/` of question, answer
   2. evaluation writes a CSV to `results/` and a plot to `results/figures`
3. View results! Pretty straightforward

Things to be aware of:

 - I had to run on Rosie for the phi models (but I only have 8gb of RAM)
