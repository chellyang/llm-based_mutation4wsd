PROMPT='''            
            You are a professional English assistant, and your task is to generate variations of the given sentence according to the specified type.
            Mutation Type:{0}
            Type Description:{1}
            Mutation follows the following requirements:
            1. Mutation must retain the main part of the original sentence as much as possible, minimizing major modifications;
            2. The mutated sentence should have practical meaning and be both grammatically and semantically correct;
            3. Mutation occurs only once in the original sentence.
            The following words in the original sentence cannot be deleted: {2}
            Mutation Example:
            1:'The cat chased the mouse.' becomes 'The mouse was chased by the cat.';
            2:'The book was written by Tom.' becomes 'Tom wrote the book.';
            Output formats are as follows:
            Mutation output format:{{"type":"Mutation Subtype","mut_sentence":"Mutated Sentence"}}
            Unchanged output format:{{"type":"","mut_sentence":""}} 
            You need to first determine whether the given original sentence can be mutated and strictly follow the json format to output the result.
            Original Sentence: {3}
            Output:
'''