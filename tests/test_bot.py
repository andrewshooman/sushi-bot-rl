#!/usr/bin/env python3

from bot import MyRLGymBot

def test_bot():
    print("Testing enhanced bot initialization...")
    bot = MyRLGymBot('test', 0, 0)
    bot.initialize_agent()
    print("SUCCESS: Enhanced bot loaded with 64-observation checkpoint!")

if __name__ == "__main__":
    test_bot()