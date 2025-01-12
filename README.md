# R2DQG

# Quick Start for Running

## Construct training data
  ```
  $ python constructSftData.py \
    --output_dir "./outputData/WQ" \
    --data_path "./data/WQ/train.json" \
    --num 3 \
    --th 0.3 \
    --train_template_path "./outputData/WQ/trainTemplates.json" \
    --gold_template_path "./outputData/WQ/gold_template_question.json" \
    --train_corrector_path "./outputData/WQ/trainCorrector.json"

  ```

## Train template generator

  ```
  $ python trainTemplateSft.py --output_dir ./outputData/WQ --train_corrector_path ./outputData/WQ/trainCorrector.json --model_path /data/rym/premodel/Meta-Llama-3-8B-Instruct/ --lora_name TemplateGenerator


  ```

## Train question corrector
  ```
  $ python trainCorrectorSft.py --output_dir ./outputData/WQ --train_corrector_path ./outputData/WQ/trainCorrector.json --model_path /data/rym/premodel/Meta-Llama-3-8B-Instruct/ --lora_name DraftCorrector

  ```

## Generate diverse questions
  ```
  $ python genInitialQuestionBasedTemplate.py \
    --output_dir "./outputData/WQ" \
    --model_name "draftQuestion" \
    --gold_template_path "./outputData/WQ/gold_template_question.json" \
    --template_path "./outputData/WQ/templates@10.json" \
    --data_path "./data/WQ/test.json" \
    --k 10 \
    --model_path "/data/rym/premodel/Meta-Llama-3-8B-Instruct/" \
    --genertor_finetune_checkpoint "TemplateGenerator/checkpoint-10000" \
    --corrector_finetune_checkpoint "DraftCorrector/checkpoint-10000"
  ```

## Evaluate diverse questions
  ```
  $ python evaluate.py \
    --output_dir "./outputData/WQ" \
    --model_name "R2DQG@10" \
    --k 10

  ```
