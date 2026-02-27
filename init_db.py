
"""
数据库初始化脚本

用于初始化对话历史数据库
"""

from src.database import get_database
from src.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """初始化数据库"""
    logger.info("=" * 60)
    logger.info("开始初始化对话数据库...")
    logger.info("=" * 60)
    
    try:
        db = get_database()
        logger.info("✅ 数据库初始化成功！")
        logger.info(f"数据库文件: conversations.db")
        logger.info("\n数据库功能:")
        logger.info("  - 对话历史存储")
        logger.info("  - 多会话支持")
        logger.info("  - 会话统计")
        logger.info("  - 会话管理（查询/删除）")
        
    except Exception as e:
        logger.error(f"❌ 数据库初始化失败: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

