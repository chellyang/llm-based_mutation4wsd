import subprocess
import os

def run_bem(dataset, res_dir):
    # python frozen_pretrained_encoder.py --data-path $path_to_wsd_data --ckpt $path_to_model_checkpoint --eval --split $wsd_eval_set
    subprocess.run(["python", "download/wsd-biencoders/finetune_pretrained_encoder.py", "--data-path",
                    "download/WSD_Evaluation_Framework", "--ckpt", "download/wsd-biencoders/wsd-biencoder",
                    "--eval", "--split", dataset])

if __name__ == '__main__':
    result_dir = "result/predictions"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    run_bem("senseval2", result_dir)