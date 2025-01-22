import pandas as pd
import time

from scraping_utils import ScrapingUtils

utils = ScrapingUtils()
# Define queries by theme
search_queries = {
    "Impact Social": [
        "mass tourism impact indigenous communities Latin America",
        "tourism effects local population South America",
        "social impact tourism Amazon communities",
        "tourism displacement local communities Latin America",
        "cultural impact mass tourism population"
    ],
    
    "Impact Économique": [
        "economic effects mass tourism Latin America local economy",
        "tourism revenue distribution South America local communities",
        "tourism jobs impact Latin America local workers",
        "tourism gentrification Latin American cities",
        "sustainable tourism economic benefits South America"
    ],
    
    "Impact Environnemental": [
        "environmental impact mass tourism Latin America",
        "tourism ecological damage South America",
        "overtourism environmental effects Amazon rainforest",
        "tourism pollution impact Andes mountains",
        "mass tourism threat biodiversity Latin America"
    ],
    
    "Solutions et Alternatives": [
        "sustainable tourism solutions Latin America",
        "community based tourism South America success",
        "ecotourism alternatives Latin America",
        "responsible tourism practices South America",
        "local tourism management Latin America case studies"
    ]
}

def flatten_dict(d, parent_key='', sep='_'):
    """
    Flattens a nested dictionary into a single-level dictionary.

    Parameters:
        d (dict): Dictionary to flatten.
        parent_key (str): Prefix for keys (used for recursion).
        sep (str): Separator to use between parent and child keys.

    Returns:
        dict: Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def flatten_data_dynamic(data):
    """
    Flattens a list of nested dictionaries into a flat DataFrame dynamically.

    Parameters:
        data (list): List of dictionaries to flatten.

    Returns:
        pd.DataFrame: Flattened DataFrame.
    """
    flattened_data = [flatten_dict(entry) for entry in data]
    return pd.DataFrame(flattened_data)

def get_multiple_queries_results(base_queries, num_results_per_query=50):
    """
    Executes multiple search queries and combines results
    
    Parameters:
        base_queries (dict): Dictionary of queries by theme
        num_results_per_query (int): Number of results per query
    Returns:
        pd.DataFrame: DataFrame.
    """
    all_results = []
    
    for theme, queries in base_queries.items():
        print(f"\nTheme search : {theme}")
        for query in queries:
            print(f"\nExecuting the query : {query}")
            df_temp = utils.get_and_scrape_articles(query, num_results=num_results_per_query)
            if not df_temp.empty:
                df_temp['theme'] = theme
                df_temp['query'] = query
                all_results.append(df_temp)
            time.sleep(5)
    
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        # URL-based duplicate removal
        final_df = final_df.drop_duplicates(subset=['url'])
        return final_df
    return pd.DataFrame()

def execute():
    # First scrap
    base_url = "https://api.worldbank.org/v2/country/all/indicator/ST.INT.ARVL"
    data = utils.get_with_api(base_url, "json", 1000)
    df = flatten_data_dynamic(data)
    df.to_csv('data_tourism_all.csv', index=False)

    # Second scrap
    df_articles = get_multiple_queries_results(search_queries, num_results_per_query=50)
    filename = 'articles_tourisme.csv'
    
    if not df_articles.empty:
        metadata = {
            'number_articles': len(df_articles),
            'number_articles_per_theme': df_articles['theme'].value_counts().to_dict()
        }
        # Save data
        df_articles.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"\nCollection completed:")
        print(f"- {metadata['number_articles']} articles saved in {filename}")
        print("\nNumber of articles per theme :")
        for theme, count in metadata['number_articles_per_theme'].items():
            print(f"- {theme}: {count}")
    else:
        print("No articles were collected.")

if __name__ == "__main__":
    execute()
