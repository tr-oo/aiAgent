from datetime import datetime, timezone
from typing import TypedDict, Sequence, Annotated
from dotenv import load_dotenv
load_dotenv()

import os
from gigachat import GigaChat
from zoneinfo import ZoneInfo
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

# Настройка GigaChat
giga = GigaChat(
    credentials=os.getenv("GIGACHAT_API_KEY"),
    verify_ssl_certs=False
)

# Определение состояния
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]

# Инструмент для получения времени
@tool
def get_current_time() -> dict:
    """Возвращает текущее UTC-время в формате ISO-8601"""
    now = datetime.utcnow().isoformat() + 'Z'
    return {"utc": now}

# Функции обработки
def call_agent(state: AgentState):
    messages = state["messages"]
    
    # Проверяем последнее сообщение
    last_message = messages[-1].content.lower()
    
    # Если запрос о времени
    if any(keyword in last_message for keyword in ["время", "time", "час", "сколько времени"]):
        return {"messages": [AIMessage(content=str(get_current_time.invoke({})))]}
    
    # Иначе используем GigaChat
    try:
        response = giga.chat(messages[-1].content)
        return {"messages": [AIMessage(content=response.choices[0].message.content)]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Ошибка: {str(e)}")]}

# Создание графа
builder = StateGraph(AgentState)
builder.add_node("agent", call_agent)
builder.set_entry_point("agent")
builder.add_edge("agent", END)

chain = builder.compile()

# Запуск сервера
if __name__ == "__main__":
    from fastapi import FastAPI
    from langserve import add_routes
    import uvicorn

    app = FastAPI()
    add_routes(app, chain, path="/time_bot")
    
    print("Сервер запущен на http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)