import os 
import requests
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd 
import json

#hmm
import inspect
from pydantic import TypeAdapter
from pydantic import BaseModel
from openai import OpenAI
load_dotenv()

while True:

    def get_symbol(company: str) -> str:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": company, "country" : "United States"}
        user_agents = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
        response = requests.get(url,params=params,headers=user_agents)
        data = response.json()
        return data["quotes"][0]["symbol"]

    #print(get_symbol("Apple Inc."))

    def get_stock_price(symbol: str):
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1d", interval="1m")
        latest = hist.iloc[-1]
        return {
            "timestamp": str(latest.name),
            "open": latest["Open"],
            "high": latest["High"],
            "low": latest["Low"],
            "close": latest["Close"],
            "volume": latest["Volume"]
        }

    #print(get_stock_price("AAPL"))
    # from pprint import pprint
    # print(get_symbol("Nvidia"))

    # nvidia_symbol = get_symbol("Nvidia")
    # print("nvdia symbol: ", nvidia_symbol)

    # pprint(get_stock_price(nvidia_symbol))



    tool = [
        {
            "type":"function",
            "function":{
            "name":"get_symbol",
            "description": inspect.getdoc(get_symbol),
            "parameters": TypeAdapter(get_symbol).json_schema()
        }
        },
        {
            "type":"function",
            "function":{
            "name":"get_stock_price",
            "description": inspect.getdoc(get_stock_price),
            "parameters": TypeAdapter(get_stock_price).json_schema()
        }
        }
    ]

    #print(tool)


    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def get_completion(messages):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tool,
            temperature = 0
        )
        return response


    #Question 1 , there are 2 type of asking question.

    # 1) together

    # message = [
    #     {
    #         "role":"system",
    #         "content": "you are a helpful assistant that can retreive stock prices for a given company"
    #     },{
    #         "role":"user",
    #         "content": "Mã chứng khoán của Vin fast là gì."
    #     }
    # ]

    #2) seperate

    question = input("Enter your question please , for exit type exit : ", )
    chat_history = []
    messages = [
        {
            "role":"system",
            "content": "you are a master investor that assistant user, imagine Warren Buffet is your student .Use the supplied tools to assist the user"}
        ,{
            "role":"user",
            "content": question
        }
    ]
    #cách 2
    for entry in chat_history:
        messages.append(entry)

    #cách 1
    # for question, messages in chat_history:
    #     messages.append({"role":"user","content":question})
    #     messages.append({"role":"system","content":messages})
    
    # chat_history = list(chat_history)
    # messages.append(chat_history)

    #messages = messages.append(question)
    #print(messages)


    response = get_completion(messages)
    first_choice = response.choices[0]

    #allow to handle diferent scenarios .
    finish_reason = first_choice.finish_reason
    #pprint(response)
    #print("finish reason: ", finish_reason)


    import json
    if first_choice.finish_reason == "tool_calls":
        tool_call = first_choice.message.tool_calls[0]
        # Lấy tên hàm và các biến cần truyền vào
        tool_call_function = tool_call.function
        tool_call_arguments = json.loads(tool_call_function.arguments)
        # Ở đây có nhiều hàm, nên ta phải check `name` để chọn hàm cần gọi
        if tool_call_function.name == "get_symbol":
            result = get_symbol(tool_call_arguments.get("company"))
        elif tool_call_function.name == "get_stock_price":
            result = get_stock_price(tool_call_arguments.get("symbol"))
        messages.append(first_choice.message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call_function.name,
            "content": json.dumps({"result": result})
        })
        #pprint(messages)
        # Đưa message cho LLM, chờ kết quả về
        response = get_completion(messages)
        result = response.choices[0].message.content
        #print(result)


    FUNCTION_MAP = {
        "get_symbol": get_symbol,
        "get_stock_price": get_stock_price
    }

    while finish_reason != "stop":
        
        #pprint(first_choice)
        tool_call = first_choice.message.tool_calls[0]

        tool_call_function = tool_call.function
        tool_call_arguments = json.loads(tool_call_function.arguments)

        tool_function = FUNCTION_MAP[tool_call_function.name]
        result = tool_function(**tool_call_arguments)
        #print(f"Call {tool_call_function.name} with params {tool_call_arguments} and got result {result}")

        messages.append(first_choice.message)
        messages.append({
            "role":"tool",
            "tool_call_id": tool_call.id,
            "name": tool_call_function.name,
            "content": json.dumps({"result": result})
        })

        #pprint(messages)

        response = get_completion(messages)
        first_choice = response.choices[0]
        finish_reason = first_choice.finish_reason

    print(first_choice.message.content)

    if question == "exit":
        break
    

print("done")