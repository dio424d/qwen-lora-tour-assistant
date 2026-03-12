from typing import Dict, Any, List, Optional

from src.amap import AmapAPI
from src.skill import Skill


class WeatherSkill(Skill):
    def __init__(self, amap: AmapAPI) -> None:
        self.amap = amap

    def name(self) -> str:
        return "weather"

    def description(self) -> str:
        return "查询城市实时天气和未来预报"

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        city = kwargs.get("city")
        if not city:
            return {"city": "", "weather": []}

        raw = self.amap.weather(city, extensions="all")
        result: Dict[str, Any] = {"city": city, "weather": []}

        if raw.get("status") != "1":
            return result

        weather_list: List[Dict[str, Any]] = []

        # 实时天气（lives）
        lives = raw.get("lives") or []
        if lives:
            live = lives[0]
            weather_list.append(
                {
                    "city": live.get("city", city),
                    "temperature": live.get("temperature"),
                    "weather": live.get("weather"),
                    "winddirection": live.get("winddirection"),
                    "windpower": live.get("windpower"),
                    "humidity": live.get("humidity"),
                    "reporttime": live.get("reporttime"),
                }
            )

        # 预报天气（forecasts）
        forecasts = raw.get("forecasts") or []
        if forecasts:
            casts = forecasts[0].get("casts", [])
            for cast in casts:
                weather_list.append(
                    {
                        "date": cast.get("date"),
                        "dayweather": cast.get("dayweather"),
                        "nightweather": cast.get("nightweather"),
                        "daytemp": cast.get("daytemp"),
                        "nighttemp": cast.get("nighttemp"),
                    }
                )

        result["weather"] = weather_list
        return result


class HotelSearchSkill(Skill):
    def __init__(self, amap: AmapAPI) -> None:
        self.amap = amap

    def name(self) -> str:
        return "hotel_search"

    def description(self) -> str:
        return "根据城市搜索酒店信息"

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        city = kwargs.get("city")
        if not city:
            return {"hotels": []}

        raw = self.amap.search_text("酒店", city)
        if raw.get("status") != "1":
            return {"hotels": []}

        pois = raw.get("pois") or []
        hotels: List[Dict[str, Any]] = []
        for poi in pois:
            hotels.append(
                {
                    "name": poi.get("name", ""),
                    "address": poi.get("address", ""),
                    "tel": poi.get("tel", ""),
                    "distance": poi.get("distance", ""),
                }
            )
        return {"hotels": hotels}


class AttractionSearchSkill(Skill):
    def __init__(self, amap: AmapAPI) -> None:
        self.amap = amap

    def name(self) -> str:
        return "attraction_search"

    def description(self) -> str:
        return "根据城市搜索旅游景点"

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        city = kwargs.get("city")
        if not city:
            return {"attractions": []}

        raw = self.amap.search_text("景点", city)
        if raw.get("status") != "1":
            return {"attractions": []}

        pois = raw.get("pois") or []
        attractions: List[Dict[str, Any]] = []
        for poi in pois:
            attractions.append(
                {
                    "name": poi.get("name", ""),
                    "address": poi.get("address", ""),
                    "tel": poi.get("tel", ""),
                    "distance": poi.get("distance", ""),
                }
            )
        return {"attractions": attractions}


class RestaurantSearchSkill(Skill):
    def __init__(self, amap: AmapAPI) -> None:
        self.amap = amap

    def name(self) -> str:
        return "restaurant_search"

    def description(self) -> str:
        return "根据城市搜索餐厅美食"

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        city = kwargs.get("city")
        if not city:
            return {"restaurants": []}

        raw = self.amap.search_text("餐厅", city)
        if raw.get("status") != "1":
            return {"restaurants": []}

        pois = raw.get("pois") or []
        restaurants: List[Dict[str, Any]] = []
        for poi in pois:
            restaurants.append(
                {
                    "name": poi.get("name", ""),
                    "address": poi.get("address", ""),
                    "tel": poi.get("tel", ""),
                    "distance": poi.get("distance", ""),
                }
            )
        return {"restaurants": restaurants}


class RoutePlanningSkill(Skill):
    def __init__(self, amap: AmapAPI) -> None:
        self.amap = amap

    def name(self) -> str:
        return "route_planning"

    def description(self) -> str:
        return "根据起点和终点进行驾车路线规划"

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        origin = kwargs.get("origin")
        destination = kwargs.get("destination")
        if not origin or not destination:
            return {"routes": []}

        # 先做地理编码
        origin_geo = self.amap.geocode(origin)
        dest_geo = self.amap.geocode(destination)

        if origin_geo.get("status") != "1" or dest_geo.get("status") != "1":
            return {"routes": []}

        try:
            origin_loc = origin_geo["geocodes"][0].get("location", "")
            dest_loc = dest_geo["geocodes"][0].get("location", "")
        except (KeyError, IndexError):
            return {"routes": []}

        raw = self.amap.direction_driving(origin_loc, dest_loc)
        if raw.get("status") != "1":
            return {"routes": []}

        paths = raw.get("route", {}).get("paths", [])
        routes: List[Dict[str, Any]] = []
        for path in paths:
            routes.append(
                {
                    "distance": path.get("distance", ""),
                    "duration": path.get("duration", ""),
                }
            )
        return {"routes": routes}


class SkillManager:
    """
    技能管理器

    - list_skills: 返回所有技能的名称和描述
    - get_skill: 按名称获取技能实例
    - find_skill: 根据用户输入自动匹配合适的技能
    """

    def __init__(self, amap: AmapAPI) -> None:
        self._skills: Dict[str, Skill] = {
            "weather": WeatherSkill(amap),
            "hotel_search": HotelSearchSkill(amap),
            "attraction_search": AttractionSearchSkill(amap),
            "restaurant_search": RestaurantSearchSkill(amap),
            "route_planning": RoutePlanningSkill(amap),
        }

    def list_skills(self) -> List[Dict[str, str]]:
        return [
            {"name": skill.name(), "description": skill.description()}
            for skill in self._skills.values()
        ]

    def get_skill(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def find_skill(self, text: str) -> Optional[Skill]:
        """
        根据用户输入的自然语言，粗略匹配对应技能。
        具体参数抽取由上层路由完成。
        """
        if not text:
            return None

        # 顺序与 README 中的技能说明保持一致
        text = text.strip()

        # 天气：用户通常会直接提到“XX天气 / 气温”
        if "天气" in text or "气温" in text:
            return self._skills.get("weather")

        # 酒店：需要同时包含“酒店/住宿”等和“找/预订/预定/预定一下”等动作词，避免普通描述就触发
        if ("酒店" in text or "住宿" in text) and any(
            kw in text for kw in ["找", "预订", "预定", "预定一下", "订", "预定个", "预订个"]
        ):
            return self._skills.get("hotel_search")

        # 景点：只在明确提到“景点”时触发，单纯说“旅游”让大模型自由发挥即可
        if "景点" in text:
            return self._skills.get("attraction_search")

        # 餐厅：避免凡是提到“吃”就触发，只在比较明确的就餐需求时调用
        if "餐厅" in text or "美食" in text or "吃饭" in text:
            return self._skills.get("restaurant_search")

        if "路线" in text or "怎么走" in text or "路怎么走" in text:
            return self._skills.get("route_planning")

        return None

