from abc import ABC, abstractmethod
from typing import Dict, Any


class Skill(ABC):
    """
    技能基类

    所有技能需要实现 name / description / execute 接口。
    """

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        执行技能

        Returns:
            统一使用 Dict[str, Any]，由上层路由进行格式化展示。
        """
        ...

