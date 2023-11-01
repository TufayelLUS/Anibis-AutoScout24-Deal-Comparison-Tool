import requests
import json
import re
import os
from urllib.parse import quote
from time import sleep
import openai
from transformers import GPT2TokenizerFast
from langchain.text_splitter import CharacterTextSplitter
import traceback
from itertools import permutations
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Libraries required:
# pip3 install requests openai transformers langchain


# configuration area
recheck_delay_in_seconds = 30  # number of seconds to wait before checking again
discount_minimum_percent = 20  # minimum discount percent to be notified
max_price_limit_anibis = 20000  # maximum price limit filter to use for anibis
# telegram bot api key
telegram_api_key = ""
telegram_chat_id = ""  # telegram target chat id
openai.api_key = ""  # openai api key
openai_query_timeout = 60 # number of seconds to wait for open ai to respond
request_timeout = 60 # number of seconds to wait for requests to complete
enable_logging = False # True = log the output/ False = no logging


# do not edit beyond this point unless you know what you're doing
makeDB = json.loads(open('makeDB.json', mode='r', encoding='utf-8').read())
mileage_values = [
    1000, 5000, 7500, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000, 180000, 190000, 200000
]
fuel_types = {
    "Essence": "268,14,245",
    "Diesel": "271,15,246",
    "Electric": "16",
    "Hybrid": "51,247,248,240,243"
}
old_memory_file = "memory.txt"
old_memory = []

all_make_model_data = open('Models.txt',
                           mode='r', encoding='utf-8').read()
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=1000, chunk_overlap=0
)
texts = text_splitter.split_text(all_make_model_data)

# if os.path.exists("output.txt"):
#     os.remove("output.txt")


def printr(text):
    print(text)
    if enable_logging:
        open('output.txt', mode='a+', encoding='utf-8').write(text + "\n")


def sendMsg(msg):
    msg = quote(msg)
    key = telegram_api_key
    msgApi = "https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}".format(
        key, telegram_chat_id, msg)
    while True:
        try:
            status = requests.get(msgApi, timeout=request_timeout).status_code
            if status == 429:
                printr(
                    "Telegram api rate limited, waiting for 1 minute to resend the alert")
                sleep(60)
            else:
                break
        except:
            printr("Could not send notification to telegram!")



def generate_combinations(input_string):
    words = input_string.split()
    combinations = []

    for i in range(1, 3):
        for perm in permutations(words, i):
            combinations.append(' '.join(perm))

    return combinations


def processWithAI(make_name, model_text, title_text):

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": " I'm going to give you a list of cars in multiple messages, which you'll have to remember throughout the conversation. The aim is for me to be able to give you names of anibis ads and for you to be able to tell me what best matches the ad in relation to the list. So from now on you'll only be able to answer with items from the list and nothing else. do you understand?"},
        {"role": "assistant", "content": "Yes, I'll remember that."}
    ]
    for text in texts:
        messages.append(
            {"role": "user", "content": "Here's the chunk of the cars list. Remember them\n" + text})
        messages.append({"role": "assistant", "content": "Noted."})
    messages.append(
        {"role": "user", "content": "Now, I'll be asking questions from here on. Answer them in this JSON format while keeping the MAKE unchanged: {\"model\": x, \"make\": y, \"version\": z}"})
    messages.append(
        {"role": "assistant", "content": "Okay, I'm ready to answer the questions now."})
    messages.append(
        {"role": "user", "content": f"Find nearest match for the given MAKE and MODEL below and identify the VERSION from the TITLE and MODEL below if exists after removing given MAKE and MODEL from the TITLE part(use your car knowledge if no match is found with my provided list earlier):\nMake: {make_name}\nModel: {model_text}\nTitle: {title_text}"})
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        request_timeout=openai_query_timeout
    )
    data = response['choices'][0]['message']['content']
    return data


def getFuelTypeForCarburant(name):
    if fuel_types.get(name):
        return fuel_types.get(name)
    return ""


def loadOldMemory():
    global old_memory
    if os.path.exists(old_memory_file):
        old_memory = open(old_memory_file, mode='r',
                          encoding='utf-8').read().split('\n')
    return old_memory


def saveMemory():
    # printr(old_memory)
    open(old_memory_file, mode='w',
         encoding='utf-8').write('\n'.join([x for x in old_memory if x.strip()]).strip() + "\n")


def isOld(link):
    global old_memory
    link = str(link)
    if link in old_memory:
        return True
    old_memory.append(link)
    return False


def findHigherMileage(given_value):
    given_value_raw = int(given_value.replace("'", '').split(' ')[0])
    for i, mileage in enumerate(mileage_values):
        if mileage == given_value_raw:
            return mileage
        elif mileage > given_value_raw:
            return mileage
    return None


def getAllPossibleVersionNames(model_name):
    model_words = model_name.split(' ')
    if len(model_words) == 1:
        return []
    model_words.pop(0)
    word_list = []
    while len(model_words):
        merged_name = " ".join(model_words)
        model_words.pop()
        word_list.append(merged_name)
    return word_list


def getMakeIDByModelName(name):
    name = name.replace(" ", "")
    for item in makeDB:
        item['text'] = item['text'].replace(" ", "")
        if item.get('text').lower() == name.lower() or item.get('text').lower().startswith(name.lower()):
            return item.get('value')
    printr("No make data found for {}".format(name))
    return ""


def getModelIDForMakeID(make_id, model_name):
    model_name = model_name.replace('-', ' ')
    link = "https://www.autoscout24.ch/webapp/v13/vehicles/models/{}?vehtyp=10".format(
        make_id)
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.193 Safari/537.36'
    }
    loaded = False
    for i in range(5):
        try:
            resp = requests.get(link, headers=headers, timeout=request_timeout).json()
            loaded = True
            break
        except:
            printr("Failed to open {}".format(link))
    if not loaded:
        return None, None
    all_models = []
    all_models_temp = resp.get('models')
    for model in all_models_temp:
        model['text'] = model['text'].replace(
            '-', ' ').encode('ascii', 'ignore').decode("utf-8")
        all_models.append(model)
    # check for exact match
    for model in all_models:
        if model.get('text').lower() == model_name.lower() or model.get('text').lower().startswith(model_name.lower()) or model_name.lower().startswith(model.get('text').lower()):
            printr("Search Model Selected: {}".format(model.get('text')))
            return model.get('value'), model.get('text')
    # check for two word partial match
    for model in all_models:
        autoscout_model_first_two_words = " ".join(
            model.get('text').split(' ')[0:2]).lower()
        anibis_model_name_first_two_words = " ".join(
            model_name.split(' ')[0:2]).lower()
        if model.get('text').lower() == anibis_model_name_first_two_words or autoscout_model_first_two_words == model_name or anibis_model_name_first_two_words == autoscout_model_first_two_words:
            printr("Search Model Selected: {}".format(model.get('text')))
            return model.get('value'), model.get('text')
    # check for one word partial match
    for model in all_models:
        autoscout_model_first_word = model.get('text').split(' ')[0].lower()
        anibis_model_first_word = model_name.split(' ')[0].lower()
        if model.get('text').lower() == anibis_model_first_word or autoscout_model_first_word == model_name or anibis_model_first_word == autoscout_model_first_word:
            printr("Search Model Selected: {}".format(model.get('text')))
            return model.get('value'), model.get('text')
    # check by removing spaces
    for model in all_models:
        no_space_a = model.get('text').lower().replace(' ', '')
        no_space_b = model_name.lower().replace(' ', '')
        if no_space_a.startswith(no_space_b) or no_space_b.startswith(no_space_a):
            printr("Search Model Selected: {}".format(model.get('text')))
            return model.get('value'), model.get('text')
    printr("No model data found for {}".format(model_name))
    return "", ""


def checkAutoScout24Results(make_id, model_id, year, mileage, gearbox, fuel_type, version_name):
    link = "https://www.autoscout24.ch/webapp/v13/vehicles"
    if gearbox.strip() == "Automatique":
        transmission = "21,209,189,188,187,190"
    else:
        transmission = "20,210,186"
    selected_mileage = findHigherMileage(mileage)
    if selected_mileage is not None:
        printr("Mileage selected for the search: {} km".format(selected_mileage))
    if version_name:
        possible_versions = generate_combinations(version_name)
        temp_stats = []
        for version in possible_versions:
            # has possible version input
            printr("Checking with version: {}".format(version))
            params = {
                'yearfrom': year,
                'kmto': str(selected_mileage),
                'trans': transmission,
                'make': make_id,
                'model': model_id,
                'typename': version,
                'sort': 'price_asc',
                'fuel': fuel_type,
                'vehtyp': '10',
            }
            if selected_mileage is None:
                del params['kmto']
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.193 Safari/537.36'
            }
            loaded = False
            for i in range(5):
                try:
                    resp = requests.get(link, headers=headers,
                                        params=params, timeout=request_timeout).json()
                    loaded = True
                    break
                except:
                    printr("Failed to open {}".format(link))
            if not loaded:
                return None
            all_vehicles = resp.get('vehicles').get('items')
            printr("Total results found: {}".format(len(all_vehicles)))
            temp_stats.append((len(all_vehicles), version, all_vehicles))
        # sort temp_stats based on first index of the tuple
        temp_stats.sort(key=lambda x: x[0], reverse=True)
        for count, version, all_vehicles in temp_stats:
            if count < 10:
                printr("Skipped version {} search due to results being {}".format(
                    version, count))
            else:
                printr("Selected final version with {} results is {}".format(
                    count, version))
                price_average = 0
                item_count = 0
                for i, vehicle in enumerate(all_vehicles[:3]):
                    item_count += 1
                    if not len(vehicle.get('prices')):
                        continue
                    item_price = vehicle.get('prices')[0].replace("'", '')
                    printr("[{}] Item: {} ({})".format(
                        i+1, vehicle.get('title'), item_price))
                    item_price_raw = re.findall(
                        r'CHF ([\d]+)\.', item_price)[0]
                    printr("Converted item price: {}".format(item_price_raw))
                    price_average += int(item_price_raw)
                if item_count > 2:
                    average_price = price_average/item_count
                    return average_price
                else:
                    printr(
                        "No average price data for this version because less than 3 results found for the given criteria")
                break
    printr("Trying without any version for this listing")
    # has no version possibility to use
    params = {
        'yearfrom': year,
        'kmto': str(selected_mileage),
        'trans': transmission,
        'make': make_id,
        'model': model_id,
        'sort': 'price_asc',
        'fuel': fuel_type,
        'vehtyp': '10',
    }
    if selected_mileage is None:
        del params['kmto']
    if fuel_type == '':
        del params['fuel']
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.193 Safari/537.36'
    }
    loaded = False
    for i in range(5):
        try:
            resp = requests.get(link, headers=headers, params=params, timeout=request_timeout).json()
            loaded = True
            break
        except:
            printr("Failed to open {}".format(link))
    if not loaded:
        return None
    all_vehicles = resp.get('vehicles').get('items')
    price_average = 0
    item_count = 0
    for i, vehicle in enumerate(all_vehicles[:3]):
        item_count += 1
        if not len(vehicle.get('prices')):
            continue
        item_price = vehicle.get('prices')[0].replace("'", '')
        printr("[{}] Item: {} ({})".format(
            i+1, vehicle.get('title'), item_price))
        item_price_raw = re.findall(r'CHF ([\d]+)\.', item_price)[0]
        printr("Converted item price: {}".format(item_price_raw))
        price_average += int(item_price_raw)
    if item_count > 2:
        average_price = price_average/item_count
        return average_price
    else:
        printr(
            "No average price data because results are less than 3 for the given criteria")
        return None


def getListingDetails(link, anibis_price):
    printr("Getting car details from {}".format(link))
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.193 Safari/537.36'
    }
    loaded = False
    for i in range(5):
        try:
            resp = requests.get(link, headers=headers, timeout=request_timeout).text
            loaded = True
            break
        except:
            printr("Failed to open {}".format(link))
    if not loaded:
        return
    page_json = resp.split('window.__INITIAL_STATE__ = ')[1].split('\n')[
        0].strip().replace('undefined', '"undefined"')
    # open('p.json', mode='w+', encoding='utf-8').write(page_json)
    page_json = json.loads(page_json)
    all_details = page_json.get('detail').get('details')
    if all_details is None:
        printr("This page may have been deleted")
        return
    make = ''
    model = ''
    year = ''
    kilometers = ''
    gearbox = ''
    carburant = ''
    title = page_json.get('detail').get('title', '')
    for detail in all_details:
        if detail.get('name') == 'Marque':
            make = detail.get('value', '').strip()
        elif detail.get('name') == 'Modèle':
            model = detail.get('value', '').strip()
        elif detail.get('name') == 'Kilomètres':
            kilometers = detail.get('value', '').strip()
        elif detail.get('name') == 'Année':
            year = detail.get('value', '').strip()
        elif detail.get('name') == 'Transmission':
            gearbox = detail.get('value', '').strip()
        elif detail.get('name') == 'Carburant':
            carburant = detail.get('value', '').strip()
    printr("Title: {}".format(title))
    printr("Make: {}".format(make))
    printr("Model: {}".format(model))
    printr("Year: {}".format(year))
    printr("Kilometers: {}".format(kilometers))
    printr("Gearbox: {}".format(gearbox))
    printr("Carburant: {}".format(carburant))
    if int(year) < 2000:
        printr("Ignored this listing as the year is below 2000")
        return
    loaded = False
    printr("Waiting for AI to generate refined response ...")
    for i in range(3):
        try:
            refined_model_n_make = processWithAI(make, model, title)
            loaded = True
            break
        except Exception as e:
            printr("Error processing with AI(retrying up to 3 times):\n{}".format(
                traceback.format_exc()))
    if not loaded:
        return
    printr("AI Response: {}".format(refined_model_n_make))
    try:
        refined_model_n_make = json.loads(refined_model_n_make)
    except:
        printr("Could not parse JSON response generated by ChatGPT, using regular method")
        refined_model_n_make = {}
    # version_possibilty = getAllPossibleVersionNames(model)
    fuel_type = getFuelTypeForCarburant(carburant)
    make_id = getMakeIDByModelName(make)
    if make_id == "":
        printr("Skipped this entry, checking next ...")
        return
    model_id, finalized_model_name = getModelIDForMakeID(
        make_id, refined_model_n_make.get('model', model))
    if len(model.split(' ')) == 1 and (refined_model_n_make.get('version') is None or refined_model_n_make.get('version') == ''):
        version_name = None
    else:
        version_name = refined_model_n_make.get('version')

    if finalized_model_name.lower() in refined_model_n_make.get('model', '').lower():
        new_version_name = refined_model_n_make.get('model', '').lower().replace(
            finalized_model_name.lower(), '').strip()
        if version_name and new_version_name != '' and not version_name.lower().startswith(new_version_name.lower()):
            version_name = new_version_name + " " + \
                version_name.lower().replace(new_version_name.lower(), '')
            printr("New version name: {}".format(version_name))
    if model_id == "":
        printr("Skipped this entry, checking next ...")
        return
    autoscout_avg = checkAutoScout24Results(
        make_id, model_id, year, kilometers, gearbox, fuel_type, version_name)
    if autoscout_avg and anibis_price:
        # calculate how much anibis price is lower than autoscout24 price in percentage
        diff = autoscout_avg - anibis_price
        diff_percentage = round(diff/autoscout_avg * 100, 2)
        printr("AutoScout24 Price In Average: {}".format(autoscout_avg))
        printr("Anibis Price: {}".format(anibis_price))
        printr("Difference: {}".format(diff))
        printr("Difference percentage: {}%".format(diff_percentage))
        if diff_percentage >= discount_minimum_percent:
            custom_msg = "Discounted Car Deal Found!\nLink: {}\nDiscount Percentage: {}%".format(
                link, diff_percentage
            )
            printr(custom_msg)
            sendMsg(custom_msg)
            printr("[!] Alert Sent")


def listOffers():
    # link = "https://api.anibis.ch/v4/fr/search/listings?aral=834__20000&cun=automobiles-voitures-de-tourisme&fcun=automobiles-voitures-de-tourisme&pr=1"
    link = "https://api.anibis.ch/v4/fr/search/listings?aral=834_5000_20000%2C833_2010_&cun=automobiles-voitures-de-tourisme&fcun=automobiles-voitures-de-tourisme&pr=1"
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.193 Safari/537.36'
    }
    loaded = False
    for i in range(5):
        try:
            resp = requests.get(link, headers=headers, timeout=request_timeout).json()
            loaded = True
            break
        except:
            printr("Failed to open {}".format(link))
    if not loaded:
        return
    all_listing = resp.get('listings')
    if all_listing is None:
        printr("No listing found")
        return
    if len(all_listing) == 0:
        printr("No listing found")
        return
    for listing in all_listing:
        listing_id = listing.get('id')
        if listing.get('url') is None:
            continue
        listing_link = "https://www.anibis.ch" + listing.get('url')
        if isOld(listing_link):
            # printr("Skipped https://www.anibis.ch{} as already processed".format(listing.get('url')))
            continue
        anibis_price = listing.get('price')
        try:
            getListingDetails(listing_link, anibis_price)
        except Exception as e:
            printr(
                "An exception occured while processing the page, trying next listing")
            traceback.print_exc()
        finally:
            printr("="*40)
            saveMemory()


if __name__ == "__main__":
    # make_id = "9"
    # model = "X5"
    # getModelIDForMakeID(make_id, model)
    # exit()
    printr("Monitoring has started ...")
    loadOldMemory()
    while True:
        printr("Checking for new listing ...")
        listOffers()
        printr("Waiting for {} seconds".format(recheck_delay_in_seconds))
        sleep(recheck_delay_in_seconds)
