This is the source code for the paper "Extensive Mutation for Testing of Word Sense
Disambiguation Models".

# **Package Requirement**

To run this code, you need to create two virtual environments and install different dependency files for them.

This is because there is a conflict in the required dependencies for the variant parts and the framework body when using zhipuAI's GLM-4.

In the following example, you first create the “mutop” virtual environment and install the dependencies in `requirements_mutop.txt`.

```bash
conda create -n mutop python=3.10
conda activate mutop
pip install -r requirements_mutop.txt
```

Then create the “zhipu” virtual environment and install the dependencies in `requirements_zhipu.txt`.

```bash
conda create -n zhipu python=3.10
conda activate zhipu
pip install -r requirements_zhipu.txt
```

python version 3.10 is used for both environments

# **Prepare dataset and models**

The data sets we use are as follows:

- Senseval-2
- Senseval-3
- Senseval-07
- Senseval-13
- Senseval-15

We provide these datasets in the `asset/WSD_Evaluation_Framework/Evaluation_Datasets/` directory, and we use its comprehensive version ALL, `asset/Evaluation_Datasets/` This directory is used to read and store the data set in our experiment, and the variation of the sentence will start from the ALL data set in this directory.

The word sense disambiguation models we tested include:

- BEM
- ESC
- EWISE
- GLOSSBERT
- SYNTAGRANK

The models BEM, ESC, EWISE, and GLOSSBERT are directly invoked locally. Ensure that the models are deployed locally correctly.

SYNTAGRANK uses api calls to ensure that the network is unblocked.

Please put the model folder under the download folder.

# WSD Testing

The code portion of the entire framework is located in the `src` folder:

- `src/prompts`: the core code of data set preprocessing, mutation, and mutation sentence screening
- `src/wsd`: indicates the defogging model. The defogging script is executed
- `src/rq`: indicates the statistics and the script for obtaining the rq result

## 1 Data Set Preprocessing

Execute `data_preprocess.py` to preprocess the ALL data set.

```bash
# Frame ontology operations need to be in the "mutop" virtual environment
conda activate mutop

python src/prompts/data_preprocess.py
```

## 2 Mutation

Execute mutate.py to mutate and generate mutated sentences.

```bash
# The steps related to calling llm must be carried out in the "zhipu" virtual environment
conda activate zhipu

# Generate original variant sentences
python src/prompts/mutate.py
```

The resulting mutated sentence is stored in the `src/prompts/preload_data/` directory.

- Variable definitions for mutation types and disambiguation systems:

```bash
mutation_types = ["antonym", "comparative", "demonstrative", "number", "passivity", "that_this", "inversion", "tenseplus", "modifier"]
wsd_systems = ["bem", "esc", "ewiser",  "glossbert", "syntagrank"]
```

- LLM configuration is located in the `src/prompts/LMmutate/config/chatglm_config.py` file, in which you need to set up by using the model and API key

```bash
# Fill in the APIKey information obtained in the console
client = ZhipuAI(api_key="your api key")  
# Used to configure the large model version
model = "glm-4"  
```

- Mutation operator prompt design in `src/prompts/LMmutate/mutations` directory, in which you can extend more mutation operator, only need to provide variable type and description (sample below)

```bash
mutation_type = "Comparative mutation"
type_description = "(1) Comparative mutation involves replacing the comparative structure of the original sentence with the superlative, or replacing the superlative structure with the comparative;" \
                   "(2) Comparative mutation does not apply to simple sentence structures."
```

## 3 Mutate Sentence Screening

The mutation results were screened using `grammer_check.py`.

```bash
# The steps related to calling llm must be carried out in the "zhipu" virtual environment
conda activate zhipu

# Generate original variant sentences
python src/prompts/LMmutate/data_helper/grammer_check.py
```

`src/mutate.py` is executed to structurally process the screened sentences and generate a dataset for model disambiguation

```bash
python src/mutate.py
```

The generated data sets for each variation type are stored with ALL in the `asset/Evaluation_Datasets/` directory.

## 4 Sense Disambiguation

Execute the disambiguation script of each disambiguation model successively to disambiguation.

```bash
source src/wsd/bem/run.sh
source src/wsd/esc/run.sh
source src/wsd/ewiser/run.sh
source src/wsd/glossbert/run.sh

# api calls for the SYNTAGRANK model
python src/wsd.py
```

Run `eval.py` to process the disambiguation results and get the disambiguation information of each target word.

```bash
python src/eval.py
```

Disambiguation results and processing results will be stored in the `result/predictions/` directory.

# **Research Questions**

The rq statistics related to the disambiguation results are located in `src/rq/`. Other RQs depend on other data, and their scripts are conveniently located elsewhere, as shown in the command execution path.

RQ1: What is the effectiveness of our mutation process?

```bash
# related processing functions: output_realistic_update(preload_list, filter_list)
python src/prompts/LMmutate/data_helper/random_30_check.py
```

RQ2: How effective is our method in reporting WSD bugs for different word sense disambiguation systems?

```bash
# related processing functions: output_bug_info(df)
python src/rq/rq.py
```

RQ3: Are the WSD bugs reported by our method genuine?

```bash
# auxiliary authenticity check
python src/rq/all_true_bugs/20_random_bug_check/check_bug_truth.py

# statistical data
python src/rq/all_true_bugs/20_random_bug_check/info.py
```

RQ4: What is the overlap between the WSD bugs reported by our mutation
operator and the baseline standards?

```bash
# related processing functions: output_all_venn(df_list_llm, df_list)
python src/rq/rq.py
```

RQ5: How is the distribution of WSD bugs discovered by our method across
different types of mutations?

```bash
# related processing functions: output_all_pie(systems_types_df_list)
python src/rq/rq.py
```

# **Discussion**

Detailed analysis of deviations from the main topic (Off-topic).

```bash
# related processing functions: output_standard_rate(mutation_types)
python src/prompts/LMmutate/data_helper/random_30_check.py
```

The ability to detect WSD bugs in different mutant types.

```bash
# related processing functions:
# output_recall_info(df), output_recall_bar(df)
python src/rq/rq.py
```

The impact of sentence complexity on mutation.

```bash
# related processing functions: output_token_relation()
python src/prompts/LMmutate/data_helper/random_30_check.py
```

# **References**

For more details about data processing, please refer to the code comments and our paper.
