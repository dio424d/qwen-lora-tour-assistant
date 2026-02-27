
import sqlite3
from datetime import datetime
from typing import List, Optional, Dict
from pathlib import Path
from src.logger import setup_logger
from src.config import settings

logger = setup_logger(__name__)


class ConversationDatabase:
    """
    对话数据库管理类
    
    使用 SQLite 存储对话历史，支持多用户、多会话
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        初始化数据库连接
        
        Args:
            db_path: 数据库文件路径，默认从配置读取
        """
        self.db_path = db_path or settings.db_path
        self._init_database()
    
    def _get_connection(self) -&gt; sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self) -&gt; None:
        """初始化数据库表结构"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id 
            ON conversations(session_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at 
            ON conversations(created_at DESC)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"数据库初始化完成: {self.db_path}")
    
    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None
    ) -&gt; int:
        """
        保存单条消息
        
        Args:
            session_id: 会话ID
            role: 角色 (user/assistant/system)
            content: 消息内容
            user_id: 用户ID（可选）
        
        Returns:
            消息ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversations (session_id, user_id, role, content)
            VALUES (?, ?, ?, ?)
        """, (session_id, user_id, role, content))
        
        conn.commit()
        message_id = cursor.lastrowid
        conn.close()
        
        logger.debug(f"保存消息: session_id={session_id}, role={role}")
        return message_id
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: int = 10
    ) -&gt; List[Dict]:
        """
        获取对话历史
        
        Args:
            session_id: 会话ID
            limit: 最多返回多少条消息
        
        Returns:
            消息列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT role, content, created_at
            FROM conversations
            WHERE session_id = ?
            ORDER BY created_at ASC
            LIMIT ?
        """, (session_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        messages = [
            {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"]
            }
            for row in rows
        ]
        
        logger.debug(f"获取对话历史: session_id={session_id}, count={len(messages)}")
        return messages
    
    def clear_session(self, session_id: str) -&gt; int:
        """
        清空会话历史
        
        Args:
            session_id: 会话ID
        
        Returns:
            删除的消息数量
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM conversations
            WHERE session_id = ?
        """, (session_id,))
        
        conn.commit()
        deleted_count = cursor.rowcount
        conn.close()
        
        logger.info(f"清空会话: session_id={session_id}, deleted={deleted_count}")
        return deleted_count
    
    def get_all_sessions(self) -&gt; List[str]:
        """
        获取所有会话ID
        
        Returns:
            会话ID列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT session_id
            FROM conversations
            ORDER BY MAX(created_at) DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        sessions = [row["session_id"] for row in rows]
        return sessions
    
    def get_session_stats(self, session_id: str) -&gt; Dict:
        """
        获取会话统计信息
        
        Args:
            session_id: 会话ID
        
        Returns:
            统计信息
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_messages,
                COUNT(CASE WHEN role = 'user' THEN 1 END) as user_messages,
                COUNT(CASE WHEN role = 'assistant' THEN 1 END) as assistant_messages,
                MIN(created_at) as first_message,
                MAX(created_at) as last_message
            FROM conversations
            WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        return {
            "total_messages": row["total_messages"],
            "user_messages": row["user_messages"],
            "assistant_messages": row["assistant_messages"],
            "first_message": row["first_message"],
            "last_message": row["last_message"]
        }


_db_instance: Optional[ConversationDatabase] = None


def get_database() -&gt; ConversationDatabase:
    """
    获取数据库单例
    
    Returns:
        ConversationDatabase 实例
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = ConversationDatabase()
    return _db_instance

