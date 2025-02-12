from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
'''article_hi = "நீ இன்று உணவு சாப்பிட்டாயா?"
article_en = "The answer is very simple."'''
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
def translation(srclang,outlang,text):
    tokenizer.src_lang = srclang
    encoded_hi = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_hi,
        forced_bos_token_id=tokenizer.lang_code_to_id[outlang]
    )
    tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    #print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
# translate Tamil to English
'''tokenizer.src_lang = "ta_IN"
encoded_hi = tokenizer(article_hi, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_hi,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
'''
# => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire dans la Syrie."

# translate eng to hin
'''tokenizer.src_lang = "en_XX"
encoded_ar = tokenizer(article_en, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_ar,
    forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "The Secretary-General of the United Nations says there is no military solution in Syria."
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))'''