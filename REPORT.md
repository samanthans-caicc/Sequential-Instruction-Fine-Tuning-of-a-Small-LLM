# Sequential Instruction Tuning of a Small Large Language Model with Strong-Model Judge Evaluation

## 1. Methodology
<details>

<ins>**Student Model**:</ins> As per recommended default model justified by its strong small-model benchmark performance, native support for the Phi-3 chat template, and practical sustainability for QLoRA-based post-training on a single 32GB V100, I decided to go ahead with the [**`Phi-3.5 Mini Instruct`**](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) model for my student model.

<ins>**Stage 1 - Alpaca Data:**</ins> The first stage of training uses `tatsu-lab/alpaca` from Starnford Alpaca that contains 52,000 examples. Out of the 52K examples, a sample size of 5,000 was drawn at random to account for HPC job time limits. Taking all 52K examples for training would take approximately 57 hours whereas 5K only took 2 hours. Samples were orgnanized into `(instruction, input, output)` formats using the Phi-3.5 chat templates. Around 250 samples were reserved for evaluation and never used for training for standard machine learning practices. These 250 samples were used to measure whether the model learned the *pattern* (followed instructions well) vs. just memorizing the training examples.

<ins>**Stage 2 - Teacher-Generated JSON:**</ins> created with imitatuion learning via Llama 3.1 8B Instruct. 70B was considered, but had to be dropped due to the 32GB VRAM limit. The pipeline consisted of:
1. five task prompt types
2. batch through teacher on Arc (UTSA)
3. validate via `json.loads` where it discarded invalids and checkpoints after every 50 for safety measures
4. store as `(instruction, input, output)`

| Task Type | n |
|---|---|
| exact_label_classification | 1,016 |
| json_extraction | 981 |
| json_repair | 974 |
| schema_constrained_generation | 942 |
| tool_call_argument_generation | 1,013 |
##### Table 1: Task type and their n values

<ins>**Hyperparameters**</ins>

| | Stage 1 | Stage 2 |
|---|---|---|
| Base | Phi-3.5 Mini Instruct | Stage 1 checkpoint |
| LoRA r / α / dropout | 16 / 32 / 0.05 | 16 / 32 / 0.05 |
| LR / scheduler | 2e-5 / cosine | 2e-5 / cosine |
| Epochs | 3 | 5 |
| Effective batch | 16 | 16 |
| Max seq length | 1024 | 2048 |
| Precision | fp16 | fp16 |
##### Table 2: Hyperparameters for stages 1 and 2

<ins>**UTSA HPC Setup:**</ins> Both stages runs on Arc `gpu4v100` (Tesla V100-32GB) via `sbatch`. Jobs were checkpointed for safety measures just in case a job could not be done at a certain point, or when the time limit (6 hours) was up. Dependenices were installed via `pip install -r requirements.txt` whereas PyTorch was separately pinned to 2.5.1+cu121 to match the CUDA 12.3 driver on V100 nodes. H100 and A100 were considered, but had to be dropped due to restrictive access and node limitations, respectively.

<ins>**Judge:**</ins> Llama 3.1 8B Instruct; same VRAM constraint as teacher where the 70B was not feasible for either case.

<ins>**Evalutation:**</ins> Three checkpoints were compared on basis of Alpaca judge win rate, Rouge-L BERTScore, JSON Validity, Scheme Compliance, and Exact Match:
1. Checkpoint 0 (base)
2. Checkpoint 1 (post stage 1)
3. Checkpoint 2 (post stage 2)

This data can be found under [Experiments](#2.-Experiments).
</details>

## 2. Experiments
<details>

### 2.1 The Three Checkpoint Comparison
#####
| Checkpoint | Alpaca Judge Win Rate | ROUGE-L / BERTScore F1 | JSON Validity | Schema Compliance | Exact Match |
|---|---|---|---|---|---|
| Ckpt 0: Untuned base | — (n=5†) | 0.158† / 0.752† | 80.0%† | 80.0%† | 0.0%† |
| Ckpt 1: After Stage 1 (Alpaca) | 60%† vs Ckpt 0 | 0.402 / 0.853 | 93.0% | 65.0% | 38.0% |
| Ckpt 2: After Stage 2 (JSON) | 40%† vs Ckpt 0 | 0.337 / 0.825 | 90.0% | 63.0% | 33.0% |
> †n=5; interpret with caution. Ckpt 0 JSON metrics unreliable at this sample size.
##### Table 3: The three checkpoint comparison
___

### 2.2 Alpaca Evaluation Results

#### **2.2.1 Pairwise win rates (judge: Llama 3.1 8B Instruct):**

| Comparison | Win | Tie | Loss | n |
|---|---|---|---|---|
| Checkpoint 1 vs Checkpoint 0 | 3 | 0 | 2 | 5† |
| Checkpoint 2 vs Checkpoint 0 | 2 | 0 | 3 | 5† |
| Checkpoint 2 vs Checkpoint 1 | 81 | 81 | 88 | 250 |
> †n=5 due to early termination of Checkpoint 0 inference job.
##### Table 4: Pairwise win rates between checkpoints
___

#### **2.2.2 Average judge scores per dimension — Alpaca eval (Ckpt 1 vs Ckpt 2, n=250):**

| Dimension | Ckpt 1 | Ckpt 2 | Delta |
|---|---|---|---|
| Instruction Following | 4.68 | 4.59 | −0.09 |
| Correctness | 4.52 | 4.52 | 0.00 |
| Clarity | 4.61 | 4.54 | −0.07 |
| Completeness | 4.54 | 4.42 | −0.12 |
| Structured Output Validity | 4.89 | 4.81 | −0.08 |
| Hallucination Risk | 4.71 | 4.74 | +0.03 |
> Ckpt 0 dimension averages omitted (n=5; too small for reliable estimates).
##### Table 5: Average judge scores per dimension via Alpaca evaluation
___

#### **2.2.3 Automatic metrics (Alpaca held-out set, n=250):**

| Metric | Ckpt 0† | Ckpt 1 | Ckpt 2 |
|---|---|---|---|
| ROUGE-L | 0.158 | 0.402 | 0.337 |
| BERTScore F1 | 0.752 | 0.853 | 0.825 |
| Avg response tokens | 220.4 | 63.0 | 46.8 |
> †n=5 only.
##### Table 6: Metrics (computed via `compute_metrics.py`


</details>

## 3. Analysis
<details>
  
</details>

## 4. Prompt Engineering
<details>

### 4.1 Teacher-Model Prompts (Imitation Learning)

### 4.2 Judge Prompts

</details>

## Appendix
<details>

### Teacher model JSON generation:
<details>
  
```
[
  {
    "task_type": "json_extraction",
    "instruction": "Extract the book title, author name, publication year, and genre from the text. Return as a JSON object with 'title', 'author', 'year', and 'genre' fields.",
    "input": "Astrophysics for People in a Hurry by Neil deGrasse Tyson was published in 2017. It belongs to the popular science genre and quickly became a bestseller."
  },
  {
    "task_type": "json_extraction",
    "instruction": "Extract all weather measurements from the meteorological report. Return a JSON object with 'temperature_celsius', 'humidity_percent', 'wind_speed_kmh', and 'conditions' fields.",
    "input": "Station report for 14:00 UTC: temperature recorded at 23.4°C, relative humidity at 67%, winds from the southwest at 18 km/h. Skies are partly cloudy."
  },
  {
    "task_type": "json_extraction",
    "instruction": "Extract the transaction details from the bank statement line. Return a JSON object with 'date', 'merchant', 'amount', and 'currency' fields.",
    "input": "2025-03-14  STRIPE*NETFLIX.COM  $15.99 USD  Debit"
  },
  {
    "task_type": "json_extraction",
    "instruction": "Extract the patient's medication list from the clinical note. Return a JSON array of objects with 'medication', 'dosage', and 'frequency' fields.",
    "input": "Patient is currently taking Metformin 500mg twice daily, Lisinopril 10mg once daily in the morning, and Atorvastatin 20mg at bedtime."
  },
  {
    "task_type": "json_extraction",
    "instruction": "Extract the technical skills mentioned in the resume excerpt and categorize them as 'programming_languages', 'frameworks', or 'tools'. Return as a JSON object with those three keys, each containing an array of strings.",
    "input": "Proficient in Python, JavaScript, and Go. Experienced with Django, React, and FastAPI. Familiar with Docker, Kubernetes, and Terraform for deployment."
  },
  {
    "task_type": "json_extraction",
    "instruction": "Extract all meeting details from the calendar invitation. Return a JSON object with 'title', 'date', 'time', 'duration_minutes', 'location', and 'attendees' fields.",
    "input": "You're invited: Q2 Budget Review on April 10, 2025 at 2:00 PM in Conference Room B (45 minutes). Attendees: Jane Smith, Tom Nguyen, Lisa Park, and Robert Chen."
  },
  {
    "task_type": "json_extraction",
    "instruction": "Parse the URL and extract its components. Return a JSON object with 'protocol', 'domain', 'port', 'path', and 'query_params' fields. Use null for absent components.",
    "input": "https://api.example.com:8443/v2/users/search?role=admin&active=true&limit=50"
  },
  {
    "task_type": "json_extraction",
    "instruction": "Extract stock market data from this ticker display. Return a JSON object with 'ticker', 'company_name', 'price', 'change', 'change_percent', and 'volume' fields. Use numbers for numeric values.",
    "input": "AAPL  Apple Inc.  $187.35  +2.14  (+1.15%)  Vol: 54.2M"
  },
  {
    "task_type": "json_extraction",
    "instruction": "Extract the recipe information from the description. Return a JSON object with 'name', 'prep_time_minutes', 'cook_time_minutes', 'servings', and 'difficulty' fields.",
    "input": "Homemade Beef Stew: This hearty recipe takes about 20 minutes to prepare and simmers for 2 hours. It serves 6 people and is rated as a medium-difficulty dish."
  },
  {
    "task_type": "json_extraction",
    "instruction": "Extract the error details from the application log line. Return a JSON object with 'timestamp', 'level', 'service', 'error_code', and 'message' fields.",
    "input": "[2025-03-01T08:45:12Z] ERROR  payment-service  ERR_GATEWAY_TIMEOUT  Downstream payment gateway did not respond within 30 seconds for transaction tx_9923abc"
  },
  {
    "task_type": "json_extraction",
    "instruction": "Extract the key terms from the contract clause. Return a JSON object with 'party_a', 'party_b', 'obligation', 'deadline', and 'penalty' fields.",
    "input": "Developer (Party A) agrees to deliver the completed API integration to ClientCorp (Party B) no later than June 1, 2025. Failure to meet this deadline incurs a penalty of $500 per business day."
  },
  {
    "task_type": "json_extraction",
    "instruction": "Extract metadata from the social media post. Return a JSON object with 'platform', 'username', 'likes', 'shares', 'timestamp', and 'hashtags' fields. Use an array for hashtags.",
    "input": "Twitter post by @techinsider at 3:22 PM · Mar 15, 2025 — 1,847 likes, 394 retweets. Hashtags: #AI #MachineLearning #OpenSource"
  },
  {
    "task_type": "json_extraction",
    "instruction": "Extract all package dependencies from the requirements snippet. Return a JSON array of objects with 'package' and 'version_constraint' fields.",
    "input": "torch>=2.1.0\ntransformers==4.45.2\npeft>=0.7.0\nbitsandbytes>=0.41.3\ndatasets~=2.18.0\ntrl>=0.8.1"
  },
  {
    "task_type": "json_extraction",
    "instruction": "Extract the commit details from the git log entry. Return a JSON object with 'hash', 'author_name', 'author_email', 'date', 'branch', and 'message' fields.",
    "input": "commit a3f8e21b  Author: Sarah Lin <sarah@devteam.io>  Date: Mon Feb 10 09:14:33 2025  Branch: feature/auth-refactor  Message: Refactor OAuth2 token refresh to handle expiry edge cases"
  },
  {
    "task_type": "json_extraction",
    "instruction": "Extract the shipping information from the label. Return a JSON object with 'tracking_number', 'carrier', 'sender_city', 'recipient_name', 'destination_city', 'destination_state', and 'service_type' fields.",
    "input": "UPS Ground  Tracking: 1Z999AA10123456784  From: Acme Warehouse, Chicago IL  To: James Okonkwo, Austin TX  Service: Ground (3-5 days)"
  },

  {
    "task_type": "schema_constrained_generation",
    "instruction": "Generate a realistic JSON object for a software engineer user profile conforming to this schema: {\"id\": string (UUID format), \"username\": string, \"email\": string, \"role\": one of \"admin\"|\"user\"|\"moderator\", \"joined_date\": string (ISO 8601 date), \"profile\": {\"full_name\": string, \"bio\": string (under 200 chars), \"skills\": array of strings (3-6 items)}}",
    "input": ""
  },
  {
    "task_type": "schema_constrained_generation",
    "instruction": "Generate a realistic JSON object for an electronics product listing matching this schema: {\"sku\": string, \"name\": string, \"brand\": string, \"category\": string, \"price_usd\": number, \"in_stock\": boolean, \"specs\": {\"weight_kg\": number, \"dimensions_cm\": {\"length\": number, \"width\": number, \"height\": number}}, \"tags\": array of strings (3-5 items)}",
    "input": ""
  },
  {
    "task_type": "schema_constrained_generation",
    "instruction": "Generate a realistic JSON API error response following RFC 7807 Problem Details format: {\"type\": string (URI), \"title\": string, \"status\": integer (HTTP status code), \"detail\": string, \"instance\": string (URI), \"errors\": array of {\"field\": string, \"message\": string} (1-3 items)}",
    "input": "Simulate a 422 Unprocessable Entity error for a user registration request with a missing required email field and an invalid phone number format."
  },
  {
    "task_type": "schema_constrained_generation",
    "instruction": "Generate a realistic JSON metadata object for a blog post matching this schema: {\"id\": string, \"title\": string, \"slug\": string (URL-safe, hyphen-separated), \"author\": {\"name\": string, \"handle\": string}, \"published_at\": string (ISO 8601), \"tags\": array of strings (3-5 tags), \"estimated_read_minutes\": integer, \"featured\": boolean}",
    "input": "The post explains transformer architecture fundamentals for beginners, written by a senior ML engineer."
  },
  {
    "task_type": "schema_constrained_generation",
    "instruction": "Generate a realistic JSON structured log entry conforming to this schema: {\"timestamp\": string (ISO 8601 with milliseconds), \"level\": one of \"DEBUG\"|\"INFO\"|\"WARN\"|\"ERROR\"|\"FATAL\", \"service\": string, \"trace_id\": string (hex, 16 chars), \"span_id\": string (hex, 8 chars), \"message\": string, \"context\": object with 2-3 key-value pairs}",
    "input": "Log an INFO message from the auth-service indicating a successful user login for user ID u_8823."
  },
  {
    "task_type": "schema_constrained_generation",
    "instruction": "Generate a realistic JSON record for a machine learning experiment conforming to this schema: {\"experiment_id\": string, \"model_name\": string, \"dataset\": string, \"hyperparams\": {\"learning_rate\": number, \"batch_size\": integer, \"epochs\": integer, \"optimizer\": string}, \"metrics\": {\"train_loss\": number, \"eval_loss\": number, \"accuracy\": number}, \"created_at\": string (ISO 8601)}",
    "input": ""
  },
  {
    "task_type": "schema_constrained_generation",
    "instruction": "Generate a realistic JSON mobile push notification payload conforming to this schema: {\"notification_id\": string (UUID), \"recipient_user_id\": string, \"title\": string (max 50 chars), \"body\": string (max 200 chars), \"type\": one of \"alert\"|\"reminder\"|\"promotion\"|\"update\", \"action_url\": string or null, \"badge_count\": integer, \"sent_at\": string (ISO 8601)}",
    "input": "Send a reminder notification about a subscription renewal due in 3 days."
  },
  {
    "task_type": "schema_constrained_generation",
    "instruction": "Generate a realistic JSON object for a restaurant menu item matching this schema: {\"item_id\": string, \"name\": string, \"description\": string, \"category\": one of \"appetizer\"|\"main\"|\"dessert\"|\"beverage\", \"price_usd\": number, \"dietary_flags\": array of zero or more of \"vegan\"|\"vegetarian\"|\"gluten_free\"|\"nut_free\", \"available\": boolean, \"prep_time_minutes\": integer}",
    "input": ""
  },
  {
    "task_type": "schema_constrained_generation",
    "instruction": "Generate a realistic JSON configuration for a CI/CD pipeline job matching this schema: {\"job_name\": string, \"trigger\": one of \"push\"|\"pull_request\"|\"schedule\"|\"manual\", \"environment\": one of \"development\"|\"staging\"|\"production\", \"steps\": array of {\"name\": string, \"command\": string} (2-4 steps), \"timeout_minutes\": integer, \"on_failure\": one of \"notify\"|\"retry\"|\"abort\"}",
    "input": "Configure a test-and-lint job that runs on pull requests to the main branch, executes flake8 then pytest, and aborts on failure."
  },
  {
    "task_type": "schema_constrained_generation",
    "instruction": "Generate a realistic JSON record for an IoT sensor reading matching this schema: {\"device_id\": string, \"sensor_type\": string, \"unit\": string, \"value\": number, \"quality\": one of \"good\"|\"degraded\"|\"bad\", \"battery_percent\": integer, \"recorded_at\": string (ISO 8601), \"location\": {\"lat\": number, \"lon\": number}}",
    "input": "A temperature sensor deployed in Austin, TX recorded a reading of 28.4 degrees Celsius."
  },
  {
    "task_type": "schema_constrained_generation",
    "instruction": "Generate a realistic JSON support ticket conforming to this schema: {\"ticket_id\": string, \"subject\": string, \"status\": one of \"open\"|\"in_progress\"|\"resolved\"|\"closed\", \"priority\": one of \"low\"|\"medium\"|\"high\"|\"critical\", \"category\": string, \"created_at\": string (ISO 8601), \"requester\": {\"name\": string, \"email\": string}, \"description\": string}",
    "input": "A user reports they cannot log in because 2FA stopped working after switching phones."
  },
  {
    "task_type": "schema_constrained_generation",
    "instruction": "Generate a realistic JSON feature flag configuration matching this schema: {\"flag_id\": string, \"name\": string, \"description\": string, \"enabled\": boolean, \"rollout_percentage\": integer (0-100), \"target_environments\": array of one or more of \"dev\"|\"staging\"|\"prod\", \"conditions\": array of {\"attribute\": string, \"operator\": one of \"eq\"|\"in\"|\"gt\"|\"lt\", \"value\": any} (0-2 conditions)}",
    "input": "Create a flag for a new dark mode UI feature, enabled for 20% of production users where the user plan is 'premium'."
  },
  {
    "task_type": "schema_constrained_generation",
    "instruction": "Generate a realistic JSON code review comment matching this schema: {\"comment_id\": string, \"file_path\": string, \"line_number\": integer, \"severity\": one of \"info\"|\"warning\"|\"error\", \"rule\": string, \"message\": string, \"suggestion\": string or null, \"auto_fixable\": boolean}",
    "input": "Flag a missing null check before calling .toString() on a potentially undefined variable in src/utils/formatter.ts at line 42."
  },
  {
    "task_type": "schema_constrained_generation",
    "instruction": "Generate a realistic JSON dataset card metadata object matching this schema: {\"dataset_id\": string, \"name\": string, \"description\": string, \"languages\": array of ISO 639-1 codes, \"license\": string (SPDX identifier), \"task_categories\": array of strings (1-3), \"size\": {\"num_examples\": integer, \"size_bytes\": integer}, \"created_by\": string, \"version\": string (semver)}",
    "input": "A multilingual sentiment analysis dataset with 50,000 English and Spanish examples, released under Apache 2.0."
  },

  {
    "task_type": "exact_label_classification",
    "instruction": "Classify the sentiment of the following text. Return a JSON object with exactly two fields: 'label' (one of exactly: 'positive', 'negative', 'neutral') and 'confidence' (float between 0.0 and 1.0, rounded to 2 decimal places).",
    "input": "The customer service was incredibly helpful and went above and beyond to resolve my issue. I'll definitely be back!"
  },
  {
    "task_type": "exact_label_classification",
    "instruction": "Classify whether the following email is spam or legitimate. Return a JSON object with 'label' (exactly 'spam' or 'ham') and 'reason' (one short sentence explaining the classification).",
    "input": "Subject: YOU HAVE BEEN SELECTED!!! Click here to claim your FREE iPhone 16 Pro. Limited offer expires TODAY. Unsubscribe: click-here.biz/unsub"
  },
  {
    "task_type": "exact_label_classification",
    "instruction": "Classify the support ticket into exactly one department. Return a JSON object with 'department' (one of exactly: 'billing', 'technical', 'account', 'shipping', 'general') and 'priority' (one of exactly: 'low', 'medium', 'high').",
    "input": "I was charged twice for my subscription this month. My bank statement shows two identical charges of $29.99 on the same date."
  },
  {
    "task_type": "exact_label_classification",
    "instruction": "Classify the severity of this code issue. Return a JSON object with 'severity' (one of exactly: 'info', 'warning', 'error', 'critical') and 'category' (one of exactly: 'style', 'performance', 'security', 'correctness', 'maintainability').",
    "input": "User input is passed directly into an SQL query string without any sanitization or parameterization."
  },
  {
    "task_type": "exact_label_classification",
    "instruction": "Classify the news headline into exactly one category. Return a JSON object with 'category' (one of exactly: 'politics', 'technology', 'sports', 'business', 'science', 'health', 'entertainment') and 'subcategory' (a single lowercase descriptive word).",
    "input": "SpaceX Successfully Launches Crew Dragon Mission to International Space Station"
  },
  {
    "task_type": "exact_label_classification",
    "instruction": "Classify the toxicity level of the social media comment. Return a JSON object with 'label' (one of exactly: 'safe', 'borderline', 'toxic'), 'categories' (array of zero or more of: 'harassment', 'hate_speech', 'profanity', 'threat'), and 'action' (one of exactly: 'allow', 'review', 'remove').",
    "input": "Great point! I hadn't thought about it from that angle. Really changes my perspective on the whole situation."
  },
  {
    "task_type": "exact_label_classification",
    "instruction": "Classify the intent of this email. Return a JSON object with 'intent' (one of exactly: 'inquiry', 'complaint', 'request', 'feedback', 'cancellation', 'subscription') and 'requires_response' (boolean).",
    "input": "Hi, I'd like to know if your API supports webhooks for real-time notifications. Also, what's the rate limit on the free tier? Thanks!"
  },
  {
    "task_type": "exact_label_classification",
    "instruction": "Based on the review text, predict the star rating. Return a JSON object with 'predicted_stars' (integer from 1 to 5) and 'key_signals' (array of exactly 3 short phrases that drove the rating).",
    "input": "Works okay for basic tasks but the battery dies way faster than advertised. The build quality feels cheap for the price. Customer support took 5 days to respond to a simple question."
  },
  {
    "task_type": "exact_label_classification",
    "instruction": "Detect the language of the text. Return a JSON object with 'language' (ISO 639-1 code), 'language_name' (English name), 'confidence' (float 0.0-1.0), and 'script' (one of exactly: 'latin', 'cyrillic', 'arabic', 'cjk', 'devanagari', 'other').",
    "input": "Die künstliche Intelligenz verändert die Art und Weise, wie wir arbeiten und leben, grundlegend."
  },
  {
    "task_type": "exact_label_classification",
    "instruction": "Classify whether this user-generated content is appropriate for a general audience. Return a JSON object with 'verdict' (one of exactly: 'approved', 'flagged', 'rejected'), 'reason' (null if approved, otherwise one short sentence), and 'confidence' (float 0.0-1.0).",
    "input": "Check out my new tutorial on how to build a REST API with FastAPI! Source code linked in bio."
  },
  {
    "task_type": "exact_label_classification",
    "instruction": "Classify the document excerpt into a type. Return a JSON object with 'document_type' (one of exactly: 'invoice', 'contract', 'report', 'email', 'press_release', 'technical_manual', 'academic_paper') and 'has_action_required' (boolean).",
    "input": "INVOICE #INV-2025-0341  To: Acme Corp  Payment due: April 15, 2025  Total: $4,750.00  Please remit payment via bank transfer to the account on file."
  },
  {
    "task_type": "exact_label_classification",
    "instruction": "Classify the log message into the appropriate log level. Return a JSON object with 'level' (one of exactly: 'DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL') and 'should_alert' (boolean).",
    "input": "Database connection pool exhausted: all 50 connections are in use. New requests are being queued. Current queue depth: 127."
  },
  {
    "task_type": "exact_label_classification",
    "instruction": "Classify the technical difficulty of this coding problem. Return a JSON object with 'difficulty' (one of exactly: 'beginner', 'intermediate', 'advanced', 'expert') and 'topics' (array of up to 3 relevant CS topic strings).",
    "input": "Implement a lock-free concurrent hash map using atomic compare-and-swap operations in C++ that supports concurrent reads and writes without deadlocks."
  },
  {
    "task_type": "exact_label_classification",
    "instruction": "Assess the pull request and classify its merge readiness. Return a JSON object with 'status' (one of exactly: 'ready', 'needs_review', 'needs_changes', 'blocked') and 'blockers' (array of strings, empty array if status is 'ready').",
    "input": "Fix typo in README. Changed 'recieve' to 'receive' in the contributing guide. No tests needed for documentation changes."
  },
  {
    "task_type": "exact_label_classification",
    "instruction": "Classify the monitoring alert. Return a JSON object with 'severity' (one of exactly: 'info', 'warning', 'critical', 'emergency'), 'service_affected' (string), 'estimated_impact' (one of exactly: 'none', 'degraded', 'partial_outage', 'full_outage'), and 'requires_immediate_action' (boolean).",
    "input": "CPU utilization on prod-api-server-03 has been above 95% for the last 8 minutes. Response times have increased by 340% and error rate is at 12%."
  },

  {
    "task_type": "json_repair",
    "instruction": "The following JSON uses single quotes instead of double quotes. Fix it to produce valid JSON.",
    "input": "{'name': 'Alice', 'age': 30, 'city': 'Austin', 'active': True}"
  },
  {
    "task_type": "json_repair",
    "instruction": "Fix the following JSON by removing trailing commas to make it valid.",
    "input": "{\"items\": [\"apple\", \"banana\", \"cherry\",], \"count\": 3, \"updated\": true,}"
  },
  {
    "task_type": "json_repair",
    "instruction": "The following JSON uses Python literals (True, False, None) instead of JSON literals (true, false, null). Convert it to valid JSON.",
    "input": "{\"active\": True, \"admin\": False, \"last_login\": None, \"score\": 98.5, \"verified\": True}"
  },
  {
    "task_type": "json_repair",
    "instruction": "Fix the following JSON-like object where keys are unquoted. Return valid JSON with all keys properly double-quoted.",
    "input": "{name: \"Bob\", age: 25, role: \"engineer\", active: true, score: 88.5}"
  },
  {
    "task_type": "json_repair",
    "instruction": "Fix the following JSON array by adding the missing comma separators between elements.",
    "input": "[{\"id\": 1, \"name\": \"Alice\"}{\"id\": 2, \"name\": \"Bob\"}{\"id\": 3, \"name\": \"Carol\"}]"
  },
  {
    "task_type": "json_repair",
    "instruction": "The following JSON was truncated mid-stream. Complete it with plausible values to produce valid, parseable JSON that preserves the existing structure.",
    "input": "{\"user\": {\"id\": \"u_4821\", \"name\": \"Jordan Lee\", \"email\": \"jordan@example.com\", \"settings\": {\"theme\": \"dark\", \"notifications\":"
  },
  {
    "task_type": "json_repair",
    "instruction": "Fix the following JSON by adding the missing closing brackets and braces to make it valid.",
    "input": "{\"order\": {\"id\": \"ord_991\", \"items\": [{\"sku\": \"A100\", \"qty\": 2}, {\"sku\": \"B200\", \"qty\": 1}"
  },
  {
    "task_type": "json_repair",
    "instruction": "In the following JSON, some numeric values are incorrectly represented as strings. Fix them so they are JSON number type (not quoted). String values that are genuinely text should remain quoted.",
    "input": "{\"temperature\": \"98.6\", \"altitude\": \"3500\", \"active\": true, \"name\": \"Sensor-A\", \"error_count\": \"0\", \"label\": \"unit-3\"}"
  },
  {
    "task_type": "json_repair",
    "instruction": "The following text contains a JSON object surrounded by non-JSON explanatory text. Extract only the JSON and return it as valid JSON without any surrounding text.",
    "input": "Sure! Here is the data you requested: {\"status\": \"success\", \"code\": 200, \"data\": {\"user\": \"admin\", \"token\": \"abc123\"}} Hope that helps!"
  },
  {
    "task_type": "json_repair",
    "instruction": "Fix the following JSON that contains a duplicate key by keeping only the last-defined value for each duplicated key. Return valid JSON.",
    "input": "{\"name\": \"Product X\", \"price\": 29.99, \"name\": \"Product X Pro\", \"category\": \"electronics\", \"price\": 49.99}"
  },
  {
    "task_type": "json_repair",
    "instruction": "The following JSON contains JavaScript-style comments which are not valid in JSON. Remove all comments (both // and /* */ styles) and return valid JSON.",
    "input": "{\n  // User configuration\n  \"theme\": \"dark\",\n  \"language\": \"en\", // default language\n  \"notifications\": true,\n  /* advanced settings */\n  \"timeout\": 30\n}"
  },
  {
    "task_type": "json_repair",
    "instruction": "Fix all errors in the following malformed JSON. It may contain unquoted keys, single quotes, Python literals, missing commas, or trailing commas. Return valid JSON.",
    "input": "{user: {'id': 42, 'permissions': ['read', 'write',], 'active': True, 'metadata': None}}"
  },
  {
    "task_type": "json_repair",
    "instruction": "The following JSON has a string value that contains an unescaped newline. Fix the escaping so the JSON is valid, representing newlines as \\n.",
    "input": "{\"message\": \"Line 1\nLine 2\nLine 3\", \"timestamp\": \"2025-01-01T00:00:00Z\"}"
  },
  {
    "task_type": "json_repair",
    "instruction": "Fix the following JSON where color values are given as bare hex strings without quotes. Return valid JSON with all hex color values properly quoted as strings.",
    "input": "{\"primary_color\": #FF5733, \"secondary_color\": #2ECC71, \"background\": #FFFFFF, \"text\": #333333}"
  },
  {
    "task_type": "json_repair",
    "instruction": "The following JSON uses integer values where boolean values are expected (1 for true, 0 for false). Convert 1 to true and 0 to false and return valid JSON.",
    "input": "{\"is_admin\": 1, \"is_verified\": 0, \"is_active\": 1, \"is_banned\": 0, \"username\": \"jsmith\", \"score\": 0}"
  },

  {
    "task_type": "tool_call_argument_generation",
    "instruction": "Generate a JSON tool call for the function 'get_current_weather' with these parameters: location (string), unit (one of 'celsius' or 'fahrenheit'), include_forecast (boolean). The 'function_name' field must also be present.",
    "input": "Get the current weather in San Antonio, Texas in Celsius and include a forecast."
  },
  {
    "task_type": "tool_call_argument_generation",
    "instruction": "Generate a JSON tool call for the function 'query_database' with these parameters: table (string), filters (object of field-value pairs), limit (integer), order_by (string), sort (one of 'asc' or 'desc'). The 'function_name' field must also be present.",
    "input": "Fetch the 10 most recently created active users from the users table, sorted by creation date descending."
  },
  {
    "task_type": "tool_call_argument_generation",
    "instruction": "Generate a JSON tool call for the function 'send_email' with these parameters: to (array of email strings), cc (array of email strings, may be empty), subject (string), body (string), priority (one of 'low', 'normal', 'high'), attachments (array of strings, may be empty). The 'function_name' field must also be present.",
    "input": "Send a high-priority meeting reminder to alice@company.com and bob@company.com, CC carol@company.com. No attachments."
  },
  {
    "task_type": "tool_call_argument_generation",
    "instruction": "Generate a JSON tool call for the function 'create_calendar_event' with these parameters: title (string), start_datetime (ISO 8601), end_datetime (ISO 8601), location (string or null), attendees (array of email strings), reminder_minutes (integer), recurring (boolean). The 'function_name' field must also be present.",
    "input": "Schedule a one-hour team standup at 9 AM on April 15, 2025 in the main conference room for eng-team@co.com. Non-recurring, 15-minute reminder."
  },
  {
    "task_type": "tool_call_argument_generation",
    "instruction": "Generate a JSON tool call for the function 'upload_file' with these parameters: source_path (string), destination_bucket (string), object_key (string), content_type (string, MIME type), is_public (boolean), metadata (object with string values, may be empty). The 'function_name' field must also be present.",
    "input": "Upload /tmp/q1_report.pdf to the S3 bucket 'company-reports' with key 'reports/2025/q1.pdf'. It should be private. No custom metadata."
  },
  {
    "task_type": "tool_call_argument_generation",
    "instruction": "Generate a JSON tool call for the function 'translate_text' with these parameters: text (string), source_language (ISO 639-1 code or 'auto'), target_language (ISO 639-1 code), preserve_formatting (boolean), formality (one of 'informal', 'formal', 'auto'). The 'function_name' field must also be present.",
    "input": "Translate 'Experience the future of smart home technology.' from English to Spanish in formal tone, preserving formatting."
  },
  {
    "task_type": "tool_call_argument_generation",
    "instruction": "Generate a JSON tool call for the function 'send_sms' with these parameters: to (string, E.164 phone format), message (string), sender_id (string), schedule_at (ISO 8601 datetime or null for immediate). The 'function_name' field must also be present.",
    "input": "Send an immediate SMS to +15125551234 from sender 'MyApp': 'Your verification code is 847291. Valid for 10 minutes.'"
  },
  {
    "task_type": "tool_call_argument_generation",
    "instruction": "Generate a JSON tool call for the function 'web_search' with these parameters: query (string), num_results (integer), search_type (one of 'web', 'news', 'images'), date_range (one of 'any', 'day', 'week', 'month', 'year'), safe_search (boolean). The 'function_name' field must also be present.",
    "input": "Search for the 5 most recent news articles about LLM fine-tuning techniques."
  },
  {
    "task_type": "tool_call_argument_generation",
    "instruction": "Generate a JSON tool call for the function 'create_charge' with these parameters: amount_cents (integer), currency (ISO 4217 code), customer_id (string), description (string), capture (boolean), metadata (object with string values, may be empty). The 'function_name' field must also be present.",
    "input": "Charge customer cus_8823xz $49.99 USD for an annual subscription renewal. Capture immediately."
  },
  {
    "task_type": "tool_call_argument_generation",
    "instruction": "Generate a JSON tool call for the function 'resize_image' with these parameters: input_path (string), output_path (string), width (integer or null), height (integer or null), maintain_aspect_ratio (boolean), output_format (one of 'jpeg', 'png', 'webp'), quality (integer 1-100 or null for lossless). The 'function_name' field must also be present.",
    "input": "Resize /images/original/banner.jpg to 1200px wide (maintain aspect ratio), save as WebP quality 85 to /images/processed/banner.webp."
  },
  {
    "task_type": "tool_call_argument_generation",
    "instruction": "Generate a JSON tool call for the function 'geocode_address' with these parameters: address (string), components (object with optional keys: country, postal_code, city), include_timezone (boolean), include_bounds (boolean). The 'function_name' field must also be present.",
    "input": "Geocode the UTSA main campus in San Antonio, TX. Include timezone, no bounds needed."
  },
  {
    "task_type": "tool_call_argument_generation",
    "instruction": "Generate a JSON tool call for the function 'execute_command' with these parameters: command (string), args (array of strings), working_directory (string), timeout_seconds (integer), capture_output (boolean), env_vars (object with string values, may be empty). The 'function_name' field must also be present.",
    "input": "Run pytest with verbose flag on the tests/ directory from /home/user/myapp with a 120-second timeout and capture output."
  },
  {
    "task_type": "tool_call_argument_generation",
    "instruction": "Generate a JSON tool call for the function 'create_github_issue' with these parameters: owner (string), repo (string), title (string), body (string), labels (array of strings), assignees (array of strings), milestone (integer or null). The 'function_name' field must also be present.",
    "input": "Create a bug report on 'samanthans-caicc/fluffy-umbrella' about inference.py crashing when the checkpoint path contains spaces. Assign to 'samanthans-caicc' with label 'bug'."
  },
  {
    "task_type": "tool_call_argument_generation",
    "instruction": "Generate a JSON tool call for the function 'push_notification' with these parameters: device_tokens (array of strings, use realistic-looking placeholder tokens), title (string), body (string), data (object, may be empty), badge (integer or null), sound (string or null), ttl_seconds (integer). The 'function_name' field must also be present.",
    "input": "Send a push notification about a new direct message to two devices. Badge count 3, default sound, 24-hour TTL."
  },
  {
    "task_type": "tool_call_argument_generation",
    "instruction": "Generate a JSON tool call for the function 'track_event' with these parameters: event_name (string), user_id (string), session_id (string), properties (object with string or number values, 2-4 properties), timestamp (ISO 8601), platform (one of 'web', 'ios', 'android', 'server'). The 'function_name' field must also be present.",
    "input": "Track a 'checkout_completed' event for user usr_2294 who purchased $89.99 on iOS."
  }
]
```
</details>

### Judge System
<details>
  
```
  You are a rigorous and impartial evaluator of language model outputs. Your task is to compare two responses (Response A and Response B) to the same instruction and score each one across multiple quality dimensions.

Rules you must follow without exception:
1. Your entire response must be valid JSON parseable by json.loads() in Python.
2. Do not include markdown code fences, explanations, or any text outside the JSON.
3. Score each dimension as an integer from 1 (very poor) to 5 (excellent).
4. For "hallucination_risk", score 5 = very low risk (trustworthy), 1 = very high risk (fabricated).
5. Declare a winner: "A", "B", or "Tie".
6. Be consistent and objective — do not favor one response based on position.

Respond only with the JSON object. Nothing else.
```
</details>

### Judge User Template
<details>

```
You are evaluating two model responses to the instruction below.
Task type: {eval_type}

## Instruction
{instruction}

## Input (if any)
{input}

## Response A
{response_a}

## Response B
{response_b}

Score each response on each dimension (1–5 integer):
- instruction_following: Did the model follow the instruction correctly?
- correctness: Is the content factually or logically correct?
- clarity: Is the response clear and well-written?
- completeness: Does the response fully address the instruction?
- structured_output_validity: Is the output correctly formatted (valid JSON for structured tasks; coherent format for general tasks)?
- hallucination_risk: How trustworthy is the response? (5 = highly trustworthy, 1 = likely fabricated)

Then declare a winner (A, B, or Tie) and provide a one-sentence justification.

Return exactly this JSON and nothing else:
{{
  "response_a_scores": {{
    "instruction_following": <int>,
    "correctness": <int>,
    "clarity": <int>,
    "completeness": <int>,
    "structured_output_validity": <int>,
    "hallucination_risk": <int>
  }},
  "response_b_scores": {{
    "instruction_following": <int>,
    "correctness": <int>,
    "clarity": <int>,
    "completeness": <int>,
    "structured_output_validity": <int>,
    "hallucination_risk": <int>
  }},
  "winner": "<A|B|Tie>",
  "justification": "<one sentence>"
}}
```
</details>

### Teacher Generation System
<details>

```
You are a precise, expert JSON generation assistant. Your task is to respond to structured-output instructions with valid, well-formatted JSON only.

Rules you must follow without exception:
1. Your entire response must be parseable as valid JSON using json.loads() in Python.
2. Do not include markdown code fences (```json or ```), explanations, or any text outside the JSON.
3. Do not add comments inside the JSON.
4. Use double quotes for all strings and keys — never single quotes.
5. Ensure all brackets, braces, and commas are correctly placed with no trailing commas.
6. If the instruction specifies a schema or required fields, follow it exactly.
7. If the instruction specifies allowed label values, use only those exact values.
8. Your response must start with either { or [ and end with the matching } or ].

Respond only with the JSON. Nothing else.
```

</details>
</details>
