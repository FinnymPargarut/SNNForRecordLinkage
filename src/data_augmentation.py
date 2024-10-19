import random
import string


def random_case_change(name):
    """Randomly change the case of the letters in the name"""
    return ''.join(random.choice([c.upper(), c.lower()]) for c in name)


def random_phone_change(phone):
    """Randomly add or remove character in phone number"""
    if random.choice([True, False]):
        phone += random.choice(string.digits)
    else:
        if len(phone) > 1:
            phone = phone[:-1]
    return phone


def random_email_change(email):
    """Randomly change letter in email"""
    if random.choice([True, False]):
        email += random.choice(string.ascii_letters)
    else:
        if len(email) > 1:
            email = email[:-1]
    return email


def augmentation(name, email, phone):
    """Applies augmentation to the data"""
    name = random_case_change(name)
    phone = random_phone_change(phone)
    email = random_email_change(email)
    return name, email, phone
