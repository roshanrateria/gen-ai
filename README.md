<div style="text-align: center; background: linear-gradient(to right, #06beb6, #48b1bf); padding: 20px; border-radius: 10px; color: white;">
    <h1 style="font-size: 2.5em; margin: 0;">Gen AI Project</h1>
    <p style="font-size: 1.2em;">A comprehensive tutorial on building, fine-tuning, and deploying lightweight, state-of-the-art NLP models using Gemma.</p>
</div>

<div style="margin: 20px 0;">
    <h2 style="color: #48b1bf;">About Gemma</h2>
    <p>Gemma is a family of lightweight, state-of-the-art open models developed using the same research and technology behind the Gemini models. These models are pre-trained on large text corpora in a self-supervised manner and are designed for a variety of NLP tasks.</p>
    <p><strong>Key Features:</strong></p>
    <ul>
        <li>High performance with low computational overhead</li>
        <li>Support for advanced fine-tuning techniques like <strong>Low Rank Adaptation (LoRA)</strong></li>
        <li>Adaptable for domain-specific NLP tasks</li>
    </ul>
</div>

<div style="margin: 20px 0;">
    <h2 style="color: #48b1bf;">Features of this Tutorial</h2>
    <ul>
        <li>Step-by-step installation and configuration of dependencies</li>
        <li>Loading and preprocessing datasets</li>
        <li>Using the <strong>GemmaCausalLM</strong> model for inference</li>
        <li>Fine-tuning the model with <strong>LoRA</strong></li>
    </ul>
</div>

<div style="margin: 20px 0;">
    <h2 style="color: #48b1bf;">Quick Start</h2>
    <h3 style="color: #06beb6;">1. Install Dependencies</h3>
    <div style="background-color: #f4f4f4; border: 1px solid #ddd; padding: 10px; border-radius: 5px; font-family: monospace;">
        <code>!pip install -q -U keras-nlp</code><br>
        <code>!pip install -q -U keras>=3</code>
    </div>

    <h3 style="color: #06beb6;">2. Set Up the Environment</h3>
    <div style="background-color: #f4f4f4; border: 1px solid #ddd; padding: 10px; border-radius: 5px; font-family: monospace;">
        <pre>
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
        </pre>
    </div>

    <h3 style="color: #06beb6;">3. Load and Preprocess the Dataset</h3>
    <div style="background-color: #f4f4f4; border: 1px solid #ddd; padding: 10px; border-radius: 5px; font-family: monospace;">
        <pre>
import json
data = []
with open('/path/to/databricks-dolly-15k.jsonl') as file:
    for line in file:
        features = json.loads(line)
        if features["context"]:
            continue
        template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
        data.append(template.format(**features))
data = data[:1000]
        </pre>
    </div>

    <h3 style="color: #06beb6;">4. Load the Model</h3>
    <div style="background-color: #f4f4f4; border: 1px solid #ddd; padding: 10px; border-radius: 5px; font-family: monospace;">
        <pre>
import keras_nlp
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
gemma_lm.summary()
        </pre>
    </div>
</div>

<div style="margin: 20px 0;">
    <h2 style="color: #48b1bf;">Fine-Tuning with LoRA</h2>
    <p>Enable Low Rank Adaptation (LoRA) to reduce trainable parameters and fine-tune the model efficiently:</p>
    <div style="background-color: #f4f4f4; border: 1px solid #ddd; padding: 10px; border-radius: 5px; font-family: monospace;">
        <pre>
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()
        </pre>
    </div>
</div>

<div style="margin: 20px 0;">
    <h2 style="color: #48b1bf;">Inference</h2>
    <p>Generate responses based on your prompt:</p>
    <div style="background-color: #f4f4f4; border: 1px solid #ddd; padding: 10px; border-radius: 5px; font-family: monospace;">
        <pre>
prompt = "What should I do on a trip to Europe?"
response = gemma_lm.generate(prompt, max_length=256)
print(response)
        </pre>
    </div>
</div>

<div style="margin: 20px 0;">
    <h2 style="color: #48b1bf;">Results and Insights</h2>
    <ul>
        <li><strong>Before fine-tuning:</strong> Basic responses that may not align well with the prompt.</li>
        <li><strong>After fine-tuning:</strong> High-quality responses tailored to the domain-specific data.</li>
    </ul>
</div>

<div style="margin: 20px 0; text-align: center; padding: 10px; background: #f9f9f9; border: 1px solid #ddd; border-radius: 5px;">
    <p>For more details, visit the <a href="https://keras.io/api/keras_nlp/models/" style="color: #48b1bf; text-decoration: none;">KerasNLP Documentation</a>.</p>
</div>
