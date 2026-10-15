"""Agent with Function Calling for Tourism Tools

Uses centralized AgentState for all execution state management.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from src.state import AgentState, initialize_state
from src.agent_core import Agent, create_agent
from src.tools import WeatherTool, AttractionsTool


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., Taitung, Taipei, 台東, 台北)"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_attractions",
            "description": "Search attractions in a city by category",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., Taitung, Taipei, 台東, 台北)"
                    },
                    "category": {
                        "type": "string",
                        "description": "Category: accommodation, activity, sightseeing, food, nature, service",
                        "enum": ["accommodation", "activity", "sightseeing", "food", "nature", "service"]
                    }
                },
                "required": ["city"]
            }
        }
    }
]


SYSTEM_PROMPT = """你是台灣觀光代理人，協助使用者查詢天氣、景點、住宿、餐飲與活動等台灣觀光資訊。

可用工具：
1. get_weather(city) - 查詢即時天氣，回傳：city, weather, weather_description, temperature, precipitation_probability, humidity, wind_speed, daily_precipitation_probability_max
2. search_attractions(city, category) - 搜尋景點/住宿/活動，回傳：name, category, district, description, indoor_outdoor, address, phone, rating

原則：
- 使用者詢問天氣 → 呼叫 get_weather
- 使用者詢問景點/住宿/活動 → 呼叫 search_attractions
- 使用者同時詢問天氣和景點 → 先查天氣，再根據天氣決定搜尋策略
- 工具回傳結構化資料，你必須在回答中引用具體數值
- 天氣回答必須包含：溫度、天氣狀況、降雨機率（若有）
- 景點回答必須列出：景點名稱、類別、所在行政區
- 不要編造工具沒有的資訊
- 回答用繁體中文"""


def create_agent(api_key: Optional[str] = None, **kwargs) -> "Agent":
    """Factory function to create agent"""
    from src.agent_core import Agent
    return Agent(api_key=api_key)


__all__ = ["Agent", "TOOLS", "SYSTEM_PROMPT", "create_agent"]