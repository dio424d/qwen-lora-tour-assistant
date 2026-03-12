import requests
from typing import Dict, Any, Optional


class AmapAPI:
    """
    高德地图 API 简单封装

    只实现项目需要的几个接口：天气、文本搜索、地理编码、驾车路线规划。
    """

    def __init__(self, api_key: str, session: Optional[requests.Session] = None, timeout: int = 8) -> None:
        if not api_key:
            raise ValueError("缺少高德地图 API 密钥")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.timeout = timeout
        self.base_url = "https://restapi.amap.com/v3"

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{path}"
        merged = {
            "key": self.api_key,
            "output": "json",
            **params,
        }
        resp = self.session.get(url, params=merged, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ---- 公开方法，与测试脚本保持兼容 ----

    def weather(self, city: str, extensions: str = "all") -> Dict[str, Any]:
        """
        天气查询
        - 默认使用 extensions=all，返回实时+预报，便于技能系统展示未来几天预报
        """
        return self._get(
            "weather/weatherInfo",
            {
                "city": city,
                "extensions": extensions,
            },
        )

    def search_text(self, keywords: str, city: str, types: Optional[str] = None) -> Dict[str, Any]:
        """
        关键字搜索（酒店/景点/餐厅等）
        """
        params: Dict[str, Any] = {
            "keywords": keywords,
            "city": city,
        }
        if types:
            params["types"] = types
        return self._get("place/text", params)

    def geocode(self, address: str) -> Dict[str, Any]:
        """
        地理编码：地址 -> 坐标
        """
        return self._get(
            "geocode/geo",
            {
                "address": address,
            },
        )

    def direction_driving(self, origin: str, destination: str) -> Dict[str, Any]:
        """
        驾车路线规划
        origin / destination 形如 "经度,纬度"
        """
        return self._get(
            "direction/driving",
            {
                "origin": origin,
                "destination": destination,
            },
        )

