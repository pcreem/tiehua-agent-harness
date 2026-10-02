# Feedback: Retry Hint Template (重試提示模板)

## 用途
當 Evaluator 判定失敗並觸發重試時，注入此模板引導 Agent 修正。

## 模板結構
```
=== 重試指引 (第 {attempt}/{max_retries} 次) ===

## 上次失敗原因
{feedback_summary}

## 具體改進方向
{actionable_feedback}

## 可用工具提醒
- get_weather(city, language): 查詢即時天氣
- search_attractions(city, category, language): 搜尋景點/住宿/活動

## 關鍵檢查清單
☐ 是否呼叫了必要的工具？
☐ 回答是否引用工具回傳的具體資料（溫度、景點名稱等）？
☐ 輸出格式是否符合要求？
☐ 語言是否與使用者一致？

## 範例修正
{correction_example}
```

## 變數說明
| 變數 | 來源 | 說明 |
|------|------|------|
| {attempt} | state.attempt | 當前重試次數 (1-indexed) |
| {max_retries} | state.max_retries | 最大重試次數 |
| {feedback_summary} | state.feedback[-1] | 最近一條評估回饋摘要 |
| {actionable_feedback} | Feedback.actionable_items | 結構化可執行建議 |
| {correction_example} | 根據 error_type 動態生成 | 針對性修正範例 |

## 針對常見錯誤的修正範例

### error_type: "missing_evidence" (未呼叫必要工具)
```
【修正範例】
上次未呼叫 get_weather 工具，導致無天氣資料可引用。
正確做法：
1. 先呼叫 get_weather(city="台東", language="zh-TW")
2. 取得溫度、降雨機率、天氣狀況
3. 回答時明確引用：ex:「今天 26°C，多雲，降雨機率 30%」
```

### error_type: "groundedness_failure" (未引用工具資料)
```
【修正範例】
上次呼叫了 search_attractions 但回答未提具體景點名稱。
正確做法：
1. 從 tool result 取得 results 陣列
2. 選擇前 3-5 筆相關結果
3. 回答時逐一引用名稱：ex:「推薦蘭舍民宿、山海民宿、晨曦民宿」
```

### error_type: "incomplete_answer" (回答為空或過短)
```
【修正範例】
上次回答過短，未提供具體建議。
正確做法：
1. 依查詢類型給出結構化回覆（天氣+景點+住宿+行動建議）
2. 每個建議說明理由
3. 主動詢問後續需求
```

### error_type: "tool_error" (工具執行失敗)
```
【修正範例】
上次 weather tool 回傳 timeout。
正確做法：
1. 重試一次（網路波動常見）
2. 若仍失敗，誠實告知使用者並提供離線建議
3. ex:「天氣服務暫時無法連線，依季節經驗台東 1 月通常 20-25°C，建議攜帶外套」
```