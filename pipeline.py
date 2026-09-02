import os
import re
import requests
from groq import Groq


class SiteFinderPipeline:
    def __init__(self):
        self.dadata_key = os.getenv("DADATA_API_KEY")
        self.brave_key = os.getenv("BRAVE_API_KEY")

        groq_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=groq_key) if groq_key else None

    def get_company_info(self, inn: str) -> dict:
        if not self.dadata_key:
            raise ValueError("DADATA_API_KEY не установлен в переменных окружения")

        url = "https://suggestions.dadata.ru/suggestions/api/4_1/rs/findById/party"
        headers = {
            "Authorization": f"Token {self.dadata_key}",
            "Content-Type": "application/json"
        }

        try:
            res = requests.post(url, json={"query": inn}, headers=headers, timeout=10).json()
            if not res.get("suggestions"):
                return None

            data = res["suggestions"][0]["data"]
            return {
                "inn": inn,
                "name": res["suggestions"][0]["value"],
                "address": data.get("address", {}).get("value", ""),
                "ogrn": data.get("ogrn", "")
            }
        except Exception as e:
            print(f"Ошибка при запросе к DaData: {e}")
            return None

    def _extract_domain(self, url: str) -> str:
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        return match.group(1).lower() if match else None

    def _call_brave_search(self, query: str, count: int = 3) -> list:
        if not self.brave_key:
            raise ValueError("BRAVE_API_KEY не установлен в переменных окружения")

        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.brave_key
        }
        params = {
            "q": query,
            "count": count
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = data.get("web", {}).get("results", [])
            return [
                {
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("description", "")
                }
                for item in results
            ]
        except Exception as e:
            print(f"Ошибка при запросе к Brave Search API: {e}")
            return []

    def search_candidates(self, company: dict) -> list:
        queries = [
            f'"{company["inn"]}"',
            f'"{company["name"]}" официальный сайт',
            f'"{company["name"]}" "{company["address"]}" сайт'
        ]

        stop_list = [
            "rusprofile.ru", "sbis.ru", "list-org.com", "spark-interfax.ru",
            "zachestnyibiznes.ru", "yandex.ru", "vk.com", "ok.ru", "tbank.ru",
            "sberbank.ru", "avito.ru", "2gis.ru", "kommersant.ru"
        ]

        candidates = []

        for q in queries:
            results = self._call_brave_search(q, count=3)
            for item in results:
                domain = self._extract_domain(item.get("url", ""))
                if domain and not any(stop in domain for stop in stop_list):
                    candidates.append({
                        "domain": domain,
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "query": q
                    })

        return candidates

    def calculate_score(self, candidate: dict, company: dict) -> float:
        score = 0.0
        text_to_check = (candidate["title"] + " " + candidate["snippet"]).lower()

        if company["inn"] in text_to_check:
            score += 0.5

        clean_name = re.sub(r'ООО|ЗАО|АО|ИП|"|\'|\s', '', company["name"]).lower()
        if clean_name and clean_name in text_to_check:
            score += 0.2

        if company["address"] and company["address"].split(",")[0].lower() in text_to_check:
            score += 0.1

        if candidate.get("freq", 1) > 1:
            score += 0.1

        return min(score, 1.0)

    def verify_with_llm(self, company: dict, candidate: dict, score: float) -> bool:
        if not self.groq_client:
            return False

        prompt = f"""
        Ты — аналитик данных. Проверь, является ли сайт официальным ресурсом организации.

        Организация: {company['name']} 
        ИНН: {company['inn']}
        Адрес: {company['address']}

        Кандидат: {candidate['domain']}
        Заголовок: {candidate['title']}
        Описание: {candidate['snippet']}
        Предварительный Score: {score}

        Ответь СТРОГО одним словом: YES (если это официальный сайт) или NO (если это агрегатор, соцсеть или посторонний сайт).
        """
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            answer = response.choices[0].message.content.strip().upper()
            return "YES" in answer
        except Exception as e:
            print(f"Ошибка вызова LLM: {e}")
            return False

    def run(self, inn: str) -> dict:
        company = self.get_company_info(inn)
        if not company:
            return {"domain": None}

        raw_candidates = self.search_candidates(company)
        if not raw_candidates:
            return {"domain": None}

        domain_map = {}
        for c in raw_candidates:
            d = c["domain"]
            if d not in domain_map:
                domain_map[d] = c
                domain_map[d]["freq"] = 1
            else:
                domain_map[d]["freq"] += 1

        best_candidate = None
        best_score = -1.0

        for d, cand in domain_map.items():
            sc = self.calculate_score(cand, company)
            if sc > best_score:
                best_score = sc
                best_candidate = cand

        if best_score >= 0.8:
            return {"domain": best_candidate["domain"]}

        elif 0.5 <= best_score < 0.8 and best_candidate:
            if self.verify_with_llm(company, best_candidate, best_score):
                return {"domain": best_candidate["domain"]}

        return {"domain": None}