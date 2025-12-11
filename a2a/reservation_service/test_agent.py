#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple test client for the reservation agent."""

import time
import requests
import sys
import io

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

AGENT_URL = "http://localhost:8001"

def chat_with_agent(prompt: str):
    """Send a message to the agent and get the response."""

    print(f"\n{'='*80}")
    print(f"YOU: {prompt}")
    print(f"{'='*80}\n")

    # Create a task
    task_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "agent.task.create",
        "params": {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        }
    }

    print("🤖 Agent is thinking...\n")

    try:
        response = requests.post(AGENT_URL, json=task_request, timeout=120)
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("\n💡 Make sure port-forward is running:")
        print("   kubectl port-forward -n team1 svc/reservation-service 8001:8000")
        return

    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return

    result = response.json()

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    # Get the task ID
    task_data = result.get("result", {})
    task_id = task_data.get("task", {}).get("id")

    if not task_id:
        print(f"❌ No task ID in response: {result}")
        return

    print(f"📋 Task created: {task_id}\n")

    # Poll for task completion
    max_attempts = 60
    for attempt in range(max_attempts):
        time.sleep(2)

        # Get task status
        status_request = {
            "jsonrpc": "2.0",
            "id": attempt + 2,
            "method": "agent.task.get",
            "params": {
                "task_id": task_id
            }
        }

        try:
            status_response = requests.post(AGENT_URL, json=status_request, timeout=30)
        except requests.exceptions.RequestException:
            continue

        if status_response.status_code != 200:
            continue

        status_result = status_response.json()
        task_info = status_result.get("result", {})

        # Print status updates
        status = task_info.get("status", "unknown")
        print(f"⏳ Status: {status}        ", end="\r")

        # Check for artifacts (final response)
        artifacts = task_info.get("artifacts", [])
        if artifacts:
            print("\n")
            for artifact in artifacts:
                parts = artifact.get("parts", [])
                for part in parts:
                    if part.get("type") == "text":
                        print(f"🤖 AGENT:\n{part.get('text', '')}\n")

            break

        # Check if completed
        if status in ["completed", "failed", "cancelled"]:
            print(f"\n✅ Task {status}")
            break

    print(f"{'='*80}\n")


def main():
    """Run the test scenarios."""

    print("\n" + "="*80)
    print(" "*25 + "RESERVATION AGENT DEMO")
    print("="*80 + "\n")

    test_prompts = [
        "Find Italian restaurants in Boston",
        "Check availability at Trattoria di Mare for 4 people on December 25th at 7 PM",
        "Make a reservation at Trattoria di Mare for December 25th at 7:00 PM, party of 4. Name: Jane Smith, Phone: +1-555-987-6543, Email: jane@example.com",
    ]

    print("This demo will showcase the restaurant reservation agent's capabilities:\n")
    print("1. 🔍 Search restaurants by city and cuisine")
    print("2. 📅 Check availability at specific restaurants")
    print("3. 📝 Make reservations with guest details\n")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'#'*80}")
        print(f"  DEMO {i}/{len(test_prompts)}")
        print(f"{'#'*80}")

        chat_with_agent(prompt)

        if i < len(test_prompts):
            print("\n⏸️  Press Enter to continue to next demo...")
            input()

    print("\n" + "="*80)
    print(" "*20 + "✅ DEMO COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
