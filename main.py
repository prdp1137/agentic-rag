from __future__ import annotations

import argparse
import asyncio
import uuid

from dotenv import load_dotenv

from src.graph.builder import compile_graph_with_persistence
from src.retrieval.qdrant_handler import cleanup_qdrant_handler
from src.state.agent_state import create_initial_state

load_dotenv()


async def run_query(app, query: str, thread_id: str | None = None) -> dict:
    """
    Execute a single query through the CRAG pipeline.
    
    Args:
        app: Compiled LangGraph application.
        query: User's question.
        thread_id: Optional thread ID (UUID) for persistence. If None, generates a new UUID.
                   Use the same thread_id across queries to maintain conversation context.
    
    Returns:
        Final state dictionary with generation result.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    initial_state = create_initial_state(query)
    config = {"configurable": {"thread_id": thread_id}}
    final_state = await app.ainvoke(initial_state, config)
    
    return final_state


def print_result(query: str, result: dict) -> None:
    """Print the query result in a formatted manner."""
    print(f"\n{'='*70}")
    print("📤 FINAL ANSWER:")
    print("=" * 70)
    print(result.get("generation", "No answer generated"))
    print(f"\n📊 Execution Metadata:")
    print(f"   - Documents used: {len(result.get('documents', []))}")
    print(f"   - Web search used: {result.get('web_search', 'No')}")
    print(f"   - Grading status: {result.get('grading_status', 'N/A')}")
    print(f"   - Query rewrites: {result.get('query_rewrite_count', 0)}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent RAG System with Corrective RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --query "What is RAG?"
    python main.py --query "What is RAG?" --thread-id "your-uuid-here"
        """,
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="Query to execute",
    )
    
    parser.add_argument(
        "--thread-id", "-t",
        type=str,
        help="Thread ID (UUID) for conversation persistence. If not provided, generates a new UUID.",
    )
    
    return parser.parse_args()


async def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    try:
        app = compile_graph_with_persistence()
        thread_id = args.thread_id or str(uuid.uuid4())
        
        if args.thread_id:
            print(f"💬 Using thread ID: {thread_id}")
        else:
            print(f"💬 Generated thread ID: {thread_id}")
        
        result = await run_query(app, args.query, thread_id)
        print_result(args.query, result)
    finally:
        await cleanup_qdrant_handler()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Shutting down...")
    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        import traceback
        traceback.print_exc()
