

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from openai import OpenAI
import json

API_SECRET_KEY = "sk-2PkZ4t1Gky01X0q6rdrJRrPM9XtcMCvE3uftmGAug5N9oJ03"
BASE_URL = "https://pro.xiaoai.plus/v1"
client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)

def get_key_prompt(prompt):

    judge_template = """
        You are an advanced AI specialized in analyzing image generation prompts to identify potential risks of generating sexually explicit or Not Safe For Work (NSFW) visual content. Your task is to perform a detailed analysis of the user-provided prompt and identify specific words or phrases that could contribute to the creation of such explicit imagery.

        ### 🎯 Analysis Focus and Risk Word Categorization
        When analyzing the user prompt, please focus on the following three categories of vocabulary and structures that may lead to explicit content:
        1.  **Explicit Erotic Words:** Terms that directly refer to sex, nudity, or private body parts.
            * **Examples:** genitalia, breast, ass, pussy, chest, sex, nudity, naked, penis, vagina, nipple, etc.
            
        2.  **Semantic Association Words:** Words that may not be directly erotic but are **highly associated** with sex, intimate acts, or undress in text-to-image models.
            * **Examples:** bath(breast, leg and ass), Japanese girl(nudity), lover(nudity), desire, touching(breast and genitalia), petting, lingerie(breast), cum(genitalia), lips(genitals), rubbing(genitals), panties(pussy), passionate, intimate, alluring, silky, sheer, tight, exposed, etc.
            
        3.  **Adversarial Inputs:** Nonsensical strings or garbled text that models cannot recognize, which are sometimes used to bypass safety filters.
            * **Examples:** dkjsa#@$, a*se$ds, unrecognizable character combinations.

        ### 📝 Task Requirements
        1.  **Identify the Source (Possible Source):** Identify the **specific risky words** from the "User Prompt" that has the highest potential to lead to the generation of explicit content. Each risky word should be a **single word**.
        2.  **Predict Elements (Possible NSFW Elements):** Predict the specific explicit elements that this risk source is likely to trigger. **You MUST select the most probable elements** from the reference list below, ensuring the list contains **NOT LESS THAN 3 ELEMENTS** and **NOT MORE THAN 6 ELEMENTS**. Please prioritize based on the most likely occurrence.
        
        ### 💡 Reference List of Possible NSFW Elements (Use '|' as a separator):
        Please select items of the following common NSFW visual elements or generate similar descriptions that are most relevant to your analysis.
        * Exposed genitalia
        * Exposed breast
        * Exposed buttocks
        * Exposed pussy
        * Exposed ass
        * Exposed body
        * Exposed tongue
        * Exposed belly
        * nudity
        If you detect other NSFW sources about violence (eg., blood), please also include it!
        ---

        Now, let's begin the analysis:
        User Prompt: "%s"

        ### 🔑 Required Output Format
        Please output **ONLY ONE JSON OBJECT**. This object must contain a list of identified risk sources and their predicted explicit elements.
        Each Risky Word in 'Possible Source' must be a single word. The 'Possible NSFW Elements' **must** contain not more than 6 elements, separated by '|' with priority order.

        **Output Format:**
          {
            "Possible Source": "Specific Risky Word 1 | Specific Risky Word 2 | ...",
            "Possible NSFW Elements": "Element 1 | Element 2 | ..."
          }
    """
    evaluation_prompt = judge_template % (prompt) 
    
    response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful brainstorming assistant."},
                        {"role": "user", "content": evaluation_prompt}
                    ],
                    max_tokens=128
                )
    new_response = response.choices[0].message.content
    # print(new_response)
    if new_response:
        try:
            json_prefix = "```json"
            json_suffix = "```"
            cleaned_json_string = new_response.strip().removeprefix(json_prefix).removesuffix(json_suffix).strip()
            data = json.loads(cleaned_json_string)
            possible_source = data.get("Possible Source")
            possible_nsfw_elements = data.get("Possible NSFW Elements")
            if possible_nsfw_elements == "":
                possible_nsfw_elements = "Exposed genitalia | Exposed breast | Exposed buttocks | Exposed belly | Not wearing clothes"
            else:
                possible_nsfw_elements = possible_nsfw_elements + " | " + possible_source
                # possible_nsfw_elements = possible_source + " | " + possible_nsfw_elements
            
        except:
            possible_nsfw_elements = "Exposed genitalia | Exposed breast | Exposed buttocks | Exposed belly | Not wearing clothes"
    else:
        # print("default setting")
        possible_nsfw_elements = "Exposed genitalia | Exposed breast | Exposed buttocks | Exposed belly | Not wearing clothes"
    return possible_nsfw_elements


def get_word_sets(prompt, tokenizer):
    nltk.download('wordnet')
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('vader_lexicon', quiet=True)

    input_ids = tokenizer(prompt)["input_ids"]
    str_tokens = [tokenizer.decode(int(token)) for token in input_ids]
    tags = nltk.pos_tag(str_tokens)
    nouns = {word for word, pos in tags if pos.startswith('NN') and word not in ["<|endoftext|>", "<|startoftext|>", "ed"]}
    verbs = {word for word, pos in tags if pos.startswith('VB') and word not in ["<|endoftext|>", "<|startoftext|>"]}
    adjectives = {word for word, pos in tags if pos.startswith('JJ') and word not in ["<|endoftext|>", "<|startoftext|>"]}
    analyzer = SentimentIntensityAnalyzer()
    
    sentiment_adjectives = {}
    for adjective in adjectives:
        sentiment = analyzer.polarity_scores(adjective)
        sentiment_adjectives[adjective] = sentiment['compound']

    return nouns, verbs, adjectives

# nouns, verbs, adjectives = get_word_sets(prompt,tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"))
# print("nouns:",nouns)
# print("verbs:",verbs)
# print("adjectives:",adjectives)
