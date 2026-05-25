import sqlite3
import os
import shutil

def reset_test_user():
    email = "test@gmail.com"
    db_path = "logs/users/users.db"
    
    if not os.path.exists(db_path):
        print("users database not found at", db_path)
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    
    if row:
        user_id = row[0]
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        print(f"Deleted user {email} (ID: {user_id}) from users.sqlite3")
        
        user_folder = f"logs/users/user_{user_id}"
        if os.path.exists(user_folder):
            ans = input(f"Are you sure you want to recursively delete the folder '{user_folder}' and all its contents? (y/n): ")
            if ans.lower() == 'y':
                # import shutil
                # shutil.rmtree(user_folder, ignore_errors=True)
                # Fallback to rm -rf if shutil.rmtree leaves the directory behind due to locks
                if os.path.exists(user_folder):
                    os.system(f"rm -rf '{user_folder}'")
                print(f"Deleted user's data folder: {user_folder}")
            else:
                print("Skipped deleting user's data folder.")
        else:
            print("No field data folder found for user.")
    else:
        print(f"User {email} not found in database.")
    
    conn.close()

if __name__ == "__main__":
    reset_test_user()
