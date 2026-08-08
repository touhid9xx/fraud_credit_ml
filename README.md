Why This Matters (Your ML Goals)
🎯 For Fraud Detection
Look at transactions table → find unusual patterns

Example: A sudden $10,000 transfer at 3 AM from an account that usually does small transactions

Connect to customer info → is this customer usually active at 3 AM?

Connect to account info → does this account normally have that much money?

🎯 For Credit Risk
Look at loans table → find who defaults on loans

Example: A customer with 5 active loans and late payments

Connect to customer info → how old are they? Are they a business or individual?

Connect to account info → do they have enough money in their accounts?

The Three "Master" Datasets We Created
To make it easy to analyze, we combined all these tables into three main datasets:

1. Transaction Dataset (48,510 rows)
What it contains: EVERY transaction with ALL the context

Who sent it (origin customer)

Who received it (destination customer)

What account it came from

What branch it happened at

What type of transaction it was
Use for: Fraud detection! Every transaction is a potential fraud case.

2. Account Dataset (1,651 rows)
What it contains: EVERY account with its COMPLETE history

Who owns it (customer info)

How many transactions it has

How many loans it has

Total money moved through it
Use for: Understanding account behavior and risk

3. Customer Dataset (1,100 rows)
What it contains: EVERY customer with their ENTIRE banking profile

How many accounts they have

Total money across all accounts

How many loans they have

Total loan amounts
Use for: Credit risk scoring and customer segmentation