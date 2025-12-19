from supabase import create_client, Client

# Supabase credentials
url = 'https://ayiwlxmvainywvpxxckn.supabase.co'
key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF5aXdseG12YWlueXd2cHh4Y2tuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg4OTE4ODUsImV4cCI6MjA2NDQ2Nzg4NX0.pRnH5b46-8PXoIG0dA3vXWr31viv-e391msqdMWnZEU'

supabase: Client = create_client(url, key)


def check_duplicate_users():
    """Check for duplicate users by email"""
    try:
        # Get all users
        result = supabase.table('users').select(
            'id, email, name, created_at').execute()
        users = result.data

        print(f"Total users found: {len(users)}")

        # Group by email to find duplicates
        email_groups = {}
        for user in users:
            email = user['email']
            if email not in email_groups:
                email_groups[email] = []
            email_groups[email].append(user)

        # Show duplicates
        duplicates = {email: users for email,
                      users in email_groups.items() if len(users) > 1}

        if duplicates:
            print(
                f"\nFound {len(duplicates)} email addresses with duplicate users:")
            for email, user_list in duplicates.items():
                print(f"\nEmail: {email}")
                for i, user in enumerate(user_list, 1):
                    print(f"  {i}. ID: {user['id']}")
                    print(f"     Name: {user['name']}")
                    print(f"     Created: {user['created_at']}")
        else:
            print("\nNo duplicate users found!")

        return duplicates

    except Exception as e:
        print(f"Error: {e}")
        return None


def check_user_languages():
    """Check user_languages table for orphaned records"""
    try:
        # Get all user_languages records
        result = supabase.table('user_languages').select('*').execute()
        user_languages = result.data

        print(f"\nTotal user_languages records: {len(user_languages)}")

        # Get all user IDs
        users_result = supabase.table('users').select('id').execute()
        valid_user_ids = {user['id'] for user in users_result.data}

        # Check for orphaned records
        orphaned_records = []
        for record in user_languages:
            if record['user_id'] not in valid_user_ids:
                orphaned_records.append(record)

        if orphaned_records:
            print(
                f"Found {len(orphaned_records)} orphaned user_languages records:")
            for record in orphaned_records:
                print(
                    f"  - ID: {record['id']}, User ID: {record['user_id']}, Language ID: {record['language_id']}")
        else:
            print("No orphaned user_languages records found!")

    except Exception as e:
        print(f"Error checking user_languages: {e}")


if __name__ == "__main__":
    print("Checking for duplicate users...")
    duplicates = check_duplicate_users()

    print("\nChecking user_languages table...")
    check_user_languages()
