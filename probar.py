import asyncio
from crawler import crawl_web
from scraper import scrape_web, top_2_data_source

# Código para probar
async def main():
    query = "arquitectura unne"
    urls_found = await crawl_web(query)
    print("URLs recuperadas:", urls_found)
    
    if urls_found:
        contents = await scrape_web(urls_found)
        print("Contenidos extraídos:")
        for content in contents:
            print('\n'+content)  # Imprime los primeros 200 caracteres
            
        print('\n\n\ncontenidos principales para formatear: ')
        print('\n'.join(line for line in top_2_data_source(contents)))

    else:
        print("No se recuperaron URLs.")

if __name__ == "__main__":
    asyncio.run(main())