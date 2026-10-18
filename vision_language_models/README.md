# Vision-Language Models: CLIP vs SmolVLM

I ran two vision-language models on the same images and tried to find where they work and where they break. One is **CLIP**, which scores an image against labels I type. The other is **SmolVLM**, a small model that answers questions about an image in plain text.

This started as a hands-on lab in my Agentic AI course at FAST-NUCES Karachi. The lab notebook came with the helper code and the test images. I ran it end to end, changed the lines marked for editing, and added the extra test cells listed below. The lab manual is not included.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Farrukh776/agentic-ai-hands-on/blob/main/vlm_lab/VLM_Lab_0776.ipynb)

## The two models

- **CLIP** (`openai/clip-vit-base-patch32`): has an image encoder and a text encoder, trained so that matching image and caption pairs end up close together. It cannot write text. It only tells me how close the image is to each label I give it. The scores go through a softmax, so they always add up to 100%.
- **SmolVLM** (`HuggingFaceTB/SmolVLM-500M-Instruct`): a vision encoder turns the image into tokens, a projector maps them into the language model, and a small LLM writes the answer one token at a time. I ask a question and it replies in free text.

I used greedy decoding (`do_sample=False`) with at most 80 new tokens, so the answers are repeatable.

## The setup

- **Hardware:** Google Colab, T4 GPU.
- **Cats image:** a COCO photo of two cats on a pink sofa, used for the CLIP and question-answering steps.
- **Generated test images:** the notebook draws its own images (circles, two shapes, text) so I always know the true answer.

## What I added

The notebook came with the models, the helper functions and the first version of each cell. These parts are mine:

- A loop that runs CLIP with three prefixes ("a photo of ", empty, "a drawing of ") on the same labels
- The "a spaceship" label test
- Three different wordings of the same question, to check if SmolVLM stays consistent
- Counting with 12 circles and a check with 3
- Two left/right layouts (square and triangle, then circle and square)
- A rotated text test with my own sentence
- My own photo, run through both models
- The written results and explanations below

## Step 1: CLIP

With labels `cat, dog, car, sofa`, CLIP put the most weight on **sofa** (68.1%) and **cat** got 31.4%. I expected cat to win, because the cats are the subject. My guess is that the sofa and blanket cover most of the picture.

**Counting with CLIP.** Labels `one cat`, `two cats`, `three cats`:

| Label | Score |
|---|---|
| two cats | 74.0% |
| three cats | 23.9% |
| one cat | 2.1% |

CLIP got this right, but "three cats" still got almost a quarter of the score, so it is not very sure.

**Changing the prefix.** Same four labels, only the words around the label change:

| Prefix | sofa | cat | car | dog |
|---|---|---|---|---|
| "a photo of " | 68.1% | 31.4% | 0.3% | 0.2% |
| "" (empty) | 73.8% | 25.6% | 0.4% | 0.2% |
| "a drawing of " | 74.2% | 24.6% | 0.9% | 0.3% |

The winner never changed, but the scores did. CLIP was trained on real captions, which look like sentences, so a bare word or a different phrase gives a slightly different text embedding.

**A silly label.** I added `a spaceship` to the list. It got 0.1%, not 0%. Softmax gives every label some share, and the scores only compete with the labels I wrote, so a wrong label can never be exactly zero.

## Step 2: SmolVLM

| Question | Answer |
|---|---|
| Describe this image in one sentence. | Two tabby cats are sleeping on a pink blanket on a couch, next to a remote control. |
| What colour is the sofa? | The sofa is pink in colour. |
| What objects are lying next to the cats? | There are remotes in the image. |

The caption is good. The sofa answer is a bit loose, because the caption says the pink thing is a blanket.

**Same question, different wording:**

| Wording | Answer |
|---|---|
| How many cats are there? | There are two cats in the image. |
| Count the cats in the picture. | Two cats are sleeping on a pink blanket. |
| Are there two cats? Answer yes or no. | Yes. |

All three agree, so for this easy image the wording did not matter.

## Step 3: Breaking the VLM

| Category | Model's answer | True answer | Result |
|---|---|---|---|
| Counting (12 circles) | 9 | 12 | **Failed** |
| Counting (3 circles) | 3 | 3 | Correct |
| Left/right (square, triangle) | The square | Square on the left | Correct |
| Left/right (circle, square) | The circle | Circle on the left | Correct |
| Text (rotated 20 degrees) | Exams are starting from next week. | Same | Correct |
| Hallucination (dog that is not there) | There is no dog present in the image. | No dog | Correct |

**Counting** was the only clear failure. It got 3 circles right and 12 wrong, which fits the idea that the model sees patches and estimates a number instead of counting each circle. I only tested 3 and 12, so I cannot say how the error grows beyond that.

**Left/right.** Both layouts were correct, even after swapping the shapes.

With only two big shapes far apart, the sides are easy to tell. It might fail on a busy image, but I did not test that.

**Reading text.** The rotated sentence was read perfectly.

The text was large and clear. I did not test a small font.

**Hallucination.** I asked "What is the dog doing in this picture?" about the cats image. A leading question like this can push a model to make something up, but SmolVLM said there is no dog. The image evidence was strong enough here.

## My own photo

I uploaded a portrait photo of a person in a dark suit. SmolVLM described a man in a dark blue suit jacket and a white shirt on a plain background, which is mostly right. It added some odd details, like a "high-low cut" jacket and a neatly folded collar, which sound normal but are not clearly in the photo. The answer also stopped in the middle of a sentence, because it hit the 80 token limit.

CLIP on the same photo, with labels `a whiteboard`, `handwritten notes`, `a desk`, `a person`:

| Label | Score |
|---|---|
| a person | 97.3% |
| a whiteboard | 2.3% |
| a desk | 0.3% |
| handwritten notes | 0.1% |

Both models got the main idea. Only SmolVLM added details that I cannot confirm. The photo is not included in this repo.

## When to use which

CLIP is the right tool when I already have a fixed list of labels and want quick classification or image search. A generative VLM is the right tool when I need open answers, captions, or questions about what is in the image. CLIP is only as good as the labels I give it, and SmolVLM can sound sure while being wrong.

## Running it

1. Open the notebook in Google Colab (the badge at the top does this).
2. Go to **Runtime > Change runtime type > T4 GPU**.
3. Run **Step 0** first. It downloads both models (about 1.6 GB together) and takes a few minutes.
4. Run the rest from top to bottom.
5. The "own photo" cell uses `google.colab.files.upload()`, so it only works inside Colab. The cats image is downloaded from the COCO website, so it needs internet.

## Notes

- Every result is from a single run with greedy decoding. I did not repeat anything, so treat small score differences (like 73.8% vs 74.2% for the prefixes) as noise.
- Counting was tested with two numbers and left/right, text and hallucination with one or two images each. These are quick checks, not a benchmark.
- SmolVLM is a very small model. Larger models would probably get more of these right, but the same kinds of failures still show up on harder cases.
- The starter code and the test image generators were provided by the course. The extra tests, the tables and the written explanations are my own.
