"""
Training Data Generator for Kenyan Phishing Detector
Helps create more training examples through templates and variations
"""

import json
import random

# Templates for generating phishing examples
PHISHING_TEMPLATES = [
    # KRA Scams
    {
        "template": "{urgency} Your KRA PIN has been {action}. {instruction} {url} to {goal} {threat}",
        "urgency": ["URGENT:", "ALERT:", "FINAL NOTICE:", "IMPORTANT:"],
        "action": ["suspended", "deactivated", "blocked", "flagged"],
        "instruction": ["Click", "Visit", "Go to", "Access"],
        "url": ["http://kra-portal.co.ke", "bit.ly/kra-verify", "kra-itax.com"],
        "goal": ["reactivate", "verify", "update", "confirm"],
        "threat": ["or face penalties", "to avoid legal action", "within 24hrs", ""],
        "label": "Phishing",
        "explanation": "Impersonates KRA, creates urgency, uses unofficial domain, threatens penalties"
    },
    # MPesa Scams
    {
        "template": "{greeting} You have {action} Ksh {amount} from {source}. {instruction} your PIN to {goal}. {urgency}",
        "greeting": ["", "Dear customer,", "Hi,"],
        "action": ["won", "received", "been credited", "qualified for"],
        "amount": ["10,000", "25,000", "50,000", "100,000"],
        "source": ["Safaricom Promotions", "MPesa Rewards", "a friend", "JAMES MWANGI"],
        "instruction": ["Send", "Reply with", "Confirm with", "Share"],
        "goal": ["claim", "accept payment", "verify transaction", "receive funds"],
        "urgency": ["Act fast!", "Limited time!", "Expires today!", ""],
        "label": "Phishing",
        "explanation": "Requests MPesa PIN, offers fake rewards, creates urgency, impersonates Safaricom"
    },
    # Bank Scams
    {
        "template": "{bank} ALERT: Your account has been {action}. {instruction} {url} {threat}",
        "bank": ["KCB BANK", "EQUITY BANK", "NCBA", "CO-OPERATIVE BANK"],
        "action": ["suspended due to suspicious activity", "locked for security", "flagged", "temporarily disabled"],
        "instruction": ["Update your details at", "Verify your account at", "Confirm identity at"],
        "url": ["http://kcb-secure.com", "equity-bank-verify.net", "ncba-login.co.ke"],
        "threat": ["or it will be closed permanently", "within 48 hours", "to restore access", ""],
        "label": "Phishing",
        "explanation": "Impersonates bank, creates panic, uses unofficial domain, threatens account closure"
    },
    # HELB Scams
    {
        "template": "HELB: Your loan of KES {amount} is {status}. {instruction} {url} {urgency}",
        "amount": ["35,000", "45,000", "60,000"],
        "status": ["ready for disbursement", "pending verification", "approved"],
        "instruction": ["Confirm details at", "Verify banking info at", "Complete application at"],
        "url": ["helb-portal.co.ke", "bit.ly/helb-form", "helb-verify.com"],
        "urgency": ["within 72hrs or funds cancelled", "before deadline", "to receive funds", ""],
        "label": "Phishing",
        "explanation": "Impersonates HELB, uses unofficial links, creates false urgency, requests banking details"
    }
]

# Templates for legitimate messages
LEGITIMATE_TEMPLATES = [
    {
        "template": "M-PESA: You have received Ksh {amount} from {sender} {phone} on {date}. New balance Ksh {balance}. Transaction cost Ksh {fee}.",
        "amount": ["500", "1,000", "2,500", "5,000"],
        "sender": ["JOHN DOE", "MARY WANJIRU", "PETER KAMAU"],
        "phone": ["0722123456", "0712345678", "0733987654"],
        "date": ["15/11/2025", "16/11/2025", "17/11/2025"],
        "balance": ["8,900.50", "12,450.00", "5,670.30"],
        "fee": ["0.00", "0.00", "0.00"],
        "label": "Legitimate",
        "explanation": "Standard MPesa receipt format, includes transaction details, no suspicious requests"
    },
    {
        "template": "Safaricom: Your {bundle} bundle is {status}. Dial *544# to buy more data. Thank you.",
        "bundle": ["data", "voice", "SMS"],
        "status": ["running low", "almost depleted", "about to expire"],
        "label": "Legitimate",
        "explanation": "Uses official Safaricom USSD code, standard notification, no suspicious links"
    }
]

def generate_message(template_dict):
    """Generate a message from template"""
    template = template_dict["template"]
    message = template
    
    # Replace placeholders
    for key, values in template_dict.items():
        if key not in ["template", "label", "explanation"]:
            placeholder = f"{{{key}}}"
            if placeholder in message:
                replacement = random.choice(values)
                message = message.replace(placeholder, replacement)
    
    return {
        "text": message,
        "label": template_dict["label"],
        "explanation": template_dict["explanation"]
    }

def generate_dataset(num_phishing=20, num_legitimate=10):
    """Generate a dataset with specified number of examples"""
    dataset = []
    
    # Generate phishing examples
    for _ in range(num_phishing):
        template = random.choice(PHISHING_TEMPLATES)
        dataset.append(generate_message(template))
    
    # Generate legitimate examples
    for _ in range(num_legitimate):
        template = random.choice(LEGITIMATE_TEMPLATES)
        dataset.append(generate_message(template))
    
    # Shuffle
    random.shuffle(dataset)
    
    return dataset

def merge_with_existing(new_data, existing_file="dataset/kenyan_phishing_data.json"):
    """Merge generated data with existing dataset"""
    try:
        with open(existing_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []
    
    # Merge and remove duplicates based on text
    all_texts = {item['text'] for item in existing_data}
    for item in new_data:
        if item['text'] not in all_texts:
            existing_data.append(item)
            all_texts.add(item['text'])
    
    return existing_data

def main():
    """Main function"""
    print("🇰🇪 Kenyan Phishing Detector - Training Data Generator")
    print("=" * 70)
    
    print("\nHow many NEW examples would you like to generate?")
    print("(This will be added to your existing dataset)")
    
    try:
        num_phishing = int(input("\nPhishing examples (recommended: 20-50): ") or "20")
        num_legitimate = int(input("Legitimate examples (recommended: 10-20): ") or "10")
    except ValueError:
        print("❌ Invalid input. Using defaults (20 phishing, 10 legitimate)")
        num_phishing = 20
        num_legitimate = 10
    
    print(f"\n🔄 Generating {num_phishing + num_legitimate} new examples...")
    
    # Generate new data
    new_data = generate_dataset(num_phishing, num_legitimate)
    
    # Show samples
    print("\n📝 Sample generated messages:")
    for i, item in enumerate(new_data[:3], 1):
        print(f"\n{i}. {item['label']}")
        print(f"   Text: {item['text'][:100]}...")
        print(f"   Explanation: {item['explanation'][:80]}...")
    
    # Ask to save
    save = input("\n💾 Save these examples? (y/n): ").lower()
    
    if save == 'y':
        # Merge with existing
        merged_data = merge_with_existing(new_data)
        
        # Save
        output_file = "dataset/kenyan_phishing_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Success!")
        print(f"   Total examples in dataset: {len(merged_data)}")
        print(f"   New examples added: {len(new_data)}")
        print(f"   File: {output_file}")
        print("\n🎯 Next steps:")
        print("   1. Review the generated examples for quality")
        print("   2. Add more manual examples if needed")
        print("   3. Retrain your model: python model/train.py")
    else:
        # Save to separate file for review
        review_file = "dataset/generated_examples_review.json"
        with open(review_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Examples saved to {review_file} for review")
        print("   Review them, then manually add good ones to your main dataset")
    
    print("\n✨ Done!")

if __name__ == "__main__":
    main()