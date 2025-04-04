#!/usr/bin/env python

from dataclasses import dataclass
from random import sample
import shelve
import click

@dataclass
class FlashCard:
    front: str
    back: str

DB_NAME = "flashcards.shlv"

@click.group()
def flashcards():
    pass

@click.command()
@click.argument("front")
@click.argument("back")
def new_flashcard (front, back):
    """Create a new flashcard
    
    FRONT is the question on the card
    BACK is the information on the back of the card
    """
    with shelve.open(DB_NAME, 'c') as shlv:
        card = FlashCard(front, back)
        cards = shlv.get("cards", [])
        cards.append(card)
        shlv["cards"] = cards

    print("New card added to flash card deck.")


@click.command()
def list_flashcards():
    """List flashcards by card number and card fronts.
    """
    with shelve.open(DB_NAME, 'c') as shlv:
        cards = shlv.get("cards", [])
        if cards:
            for index, card in enumerate(cards):
                print(f"{index + 1}. {card.front}")
        else:
            print("No cards in deck currently")

@click.command()
@click.argument("card_number", type=int)
def delete_flashcard(card_number):
    """Delete a flashcard

    CARD_NUMBER is the number displayed before the question when running the `list` command
    """
    with shelve.open(DB_NAME, 'c') as shlv:
        cards = shlv.get("cards", [])
        if card_number > 0 and card_number <= len(cards):
            card = cards.pop(card_number - 1)
            shlv["cards"] = cards
            print(f"Card Deleted: {card.front}")
            #for index, card in enumerate(cards):
            #    card = cards.pop(card_number - 1)
            #    shlv["cards"] = cards
            #    print(f"Card Deleted: {card.front}")
        else:
            print(f"Unable to delete card number {card_number}")

@click.command()
@click.argument("number_of_cards", default=10, type=int)
def practice(number_of_cards):
    """Present flashcards to study

    NUMBER_OF_CARDS is the number of questions to ask. Default to 10
    """
    with shelve.open(DB_NAME, 'c') as shlv:
        cards = shlv.get("cards", [])
        if not cards:
            print("No flashcards created yet.")
            return
           
        if number_of_cards > len(cards):
            number_of_cards = len(cards)
        
        print(f"Asking {number_of_cards} questions.")
        questions_to_ask = sample(cards, number_of_cards)
        for index, question in enumerate(questions_to_ask):
            input(
                f"Question {index + 1}. (presss Enter to reveal the back of card):\n\n{question.front}\n"
            )
            print(question.back)
            if not index + 1 == number_of_cards:
                input("\nPress Enter to see the next question.")


flashcards.add_command(new_flashcard, "new")
flashcards.add_command(list_flashcards, "list")
flashcards.add_command(delete_flashcard, "delete")
flashcards.add_command(practice, "practice")

if __name__ == "__main__":
    flashcards()
