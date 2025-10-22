def valid_date(date):
    """You have to write a function which validates a given date string and
    returns True if the date is valid otherwise False.
    The date is valid if all of the following rules are satisfied:
    1. The date string is not empty.
    2. The number of days is not less than 1 or higher than 31 days for months 1,3,5,7,8,10,12. And the number of days is not less than 1 or higher than 30 days for months 4,6,9,11. And, the number of days is not less than 1 or higher than 29 for the month 2.
    3. The months should not be less than 1 or higher than 12.
    4. The date should be in the format: mm-dd-yyyy

    for example: 
    valid_date('03-11-2000') => True

    valid_date('15-01-2012') => False

    valid_date('04-0-2040') => False

    valid_date('06-04-2020') => True

    valid_date('06/04/2020') => False
    """
    if not date:
        return False

    parts = date.split('-')
    if len(parts) != 3:
        return False

    try:
        month = int(parts[0])
        day = int(parts[1])
        year = int(parts[2])
    except ValueError:
        return False

    # Rule 3: The months should not be less than 1 or higher than 12.
    if not (1 <= month <= 12):
        return False

    # Rule 2: Validate days based on month
    if month in [1, 3, 5, 7, 8, 10, 12]:
        if not (1 <= day <= 31):
            return False
    elif month in [4, 6, 9, 11]:
        if not (1 <= day <= 30):
            return False
    elif month == 2:
        # For simplicity, the problem statement says "not higher than 29 for the month 2".
        # It doesn't explicitly mention leap years, so we stick to the stated rule.
        if not (1 <= day <= 29):
            return False
    else:
        # This case should ideally not be reached if month validation (Rule 3) is correct,
        # but as a safeguard.
        return False

    return True