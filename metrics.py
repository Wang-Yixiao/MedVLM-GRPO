import evaluate
def compute_metrics_test(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
    
def compute_metrics(eval_pred):
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result_rouge = {key: value * 100 for key, value in result_rouge.items()}

    result_bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result_bleu = {key: value * 100 for key, value in result_bleu.items()}
 
    return {k: round(v, 4) for k, v in result_rouge.items()}

