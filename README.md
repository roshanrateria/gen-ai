<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Gen AI Project</h1>

<div class="highlight">
    <strong>Welcome to the Gen AI Project</strong>: A comprehensive tutorial on building, fine-tuning, and deploying lightweight, state-of-the-art NLP models using Gemma.
</div>

<h2>About Gemma</h2>
<p>Gemma is a family of lightweight, state-of-the-art open models developed using the same research and technology behind the Gemini models. These models are designed for various NLP tasks and are pre-trained on large text corpora in a self-supervised manner.</p>
<p>Gemma models offer:</p>
<ul>
    <li>High performance with low computational overhead</li>
    <li>Advanced fine-tuning techniques like <strong>Low Rank Adaptation (LoRA)</strong></li>
    <li>Support for domain-specific tasks</li>
</ul>

<h2>Features of this Tutorial</h2>
<ul class="steps">
    <li>Install and configure dependencies</li>
    <li>Load and preprocess datasets</li>
    <li>Use the <strong>GemmaCausalLM</strong> model for inference</li>
    <li>Fine-tune the model using <strong>LoRA</strong> techniques</li>
</ul>

<h2>Getting Started</h2>

<h3>Install Dependencies</h3>
<div class="code-block">
    <pre>
!pip install -q -U keras-nlp
!pip install -q -U keras>=3
    </pre>
</div>
<p>Ensure that Keras and KerasNLP are up-to-date for compatibility.</p>

<h3>Set Up the Environment</h3>
<div class="code-block">
    <pre>
import os

os.environ["KERAS_BACKEND"] = "jax" 
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
    </pre>
</div>

<h3>Load and Preprocess the Dataset</h3>
<p>We are using a subset of 1000 training examples from the <a href="https://www.kaggle.com/datasets">Databricks Dolly 15K Dataset</a>.</p>
<div class="code-block">
    <pre>
import json
data = []
with open('/path/to/databricks-dolly-15k.jsonl') as file:
    for line in file:
        features = json.loads(line)
        if features["context"]:
            continue
        template = "Instruction:\\n{instruction}\\n\\nResponse:\\n{response}"
        data.append(template.format(**features))
data = data[:1000]
    </pre>
</div>

<h3>Load the Model</h3>
<p>We are using the <strong>GemmaCausalLM</strong> model for causal language modeling:</p>
<div class="code-block">
    <pre>
import keras_nlp

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
gemma_lm.summary()
    </pre>
</div>

<h2>Fine-Tuning with LoRA</h2>
<p>Fine-tune the model using Low Rank Adaptation (LoRA) to reduce the number of trainable parameters:</p>
<div class="code-block">
    <pre>
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()
    </pre>
</div>
<p>Note: LoRA reduces trainable parameters significantly, making the model more efficient.</p>

<h2>Inference</h2>
<p>Generate responses based on your prompts:</p>
<div class="code-block">
    <pre>
prompt = template.format(
    instruction="What should I do on a trip to Europe?",
    response="",
)
print(gemma_lm.generate(prompt, max_length=256))
    </pre>
</div>

<h2>Results</h2>
<p>The model provides detailed and accurate responses after fine-tuning. For example:</p>
<ul>
    <li><strong>Europe Trip Prompt:</strong> Suggestions for famous sights and itinerary planning.</li>
    <li><strong>Photosynthesis Prompt:</strong> Simplified explanations suitable for children.</li>
</ul>

<h2>Conclusion</h2>
<p>This tutorial demonstrates how to effectively use Gemma models for NLP tasks. By leveraging LoRA for fine-tuning, we achieve high-quality results with minimal computational resources.</p>

<h2>Further Reading</h2>
<p>For more details, check out the <a href="https://keras.io/api/keras_nlp/models/">KerasNLP documentation</a>.</p>

<footer>
    <p>Created by: <a href="https://github.com/roshanrateria">Roshan Rateria</a> | Repository: <a href="https://github.com/roshanrateria/gen-ai">Gen AI Project</a></p>
</footer>

</body>
</html>
