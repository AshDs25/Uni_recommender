import sqlite3
import json

def fetch(universities_names):

    conn = sqlite3.connect(r'D:\Uni_Recommender\Unixplore.db')

    # Connect to the SQLite database
    cursor = conn.cursor()
    cursor2 = conn.cursor()
    # Create a dictionary to store the university data
    universities_dict = {}

    # Iterate over the list of university names
    for uni_name in universities_names:
        
        # Fetch additional data for each university using separate queries
        query = "SELECT Uni_desc, Uni_acceptance, Uni_test,Uni_min_test_score ,Uni_image FROM University WHERE University_name='{}'".format(uni_name)
        cursor.execute(query)
        uni_data = cursor.fetchone()
        
        query2 = "SELECT Uni_courses,Uni_stream,Uni_co_dur,Uni_fees FROM University WHERE University_name='{}'".format(uni_name)
        cursor2.execute(query2)
        uni_courses = cursor2.fetchall()
        
        # Create a dictionary to store the additional data for the university
        additional_data = {
            'description': uni_data[0],  # Assuming the first column is Uni_desc
            'acceptance_rate': uni_data[1],  # Assuming the second column is Uni_acceptance_rate
            'test': uni_data[2],  # Assuming the third column is Uni_test_score
            'test_score':uni_data[3],
            'image': uni_data[4],  # Assuming the fourth column is Uni_image
            'courses':[course[0] for course in uni_courses],
            'stream':[stream[1] for stream in uni_courses],
            'duration':[duration[2] for duration in uni_courses],
            'fees':[fees[3].replace('?','₹') for fees in uni_courses]
        }
        
        # Add the additional data to the dictionary for the university
        universities_dict[uni_name] = additional_data
        


    # Close the cursor and connection
    cursor.close()
    cursor2.close()
    conn.close()

    json_string = json.dumps(universities_dict)
    return json_string
