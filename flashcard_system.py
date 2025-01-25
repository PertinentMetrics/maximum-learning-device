from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Deque, Dict, Tuple
from collections import deque, defaultdict
import pickle
from pathlib import Path
import time
import customtkinter as ctk

@dataclass
class Flashcard:
    question: str
    answer: str
    review_level: int = 1
    seconds_until_review: float = 0.0
    success_rate: float = 0.0
    times_correct: int = 0
    reviews: int = 0
    is_new_flashcard: bool = True
    time_of_last_review: float = time.time()

    def increment_review_level(self) -> None:
        """Increment the review level and update stats"""
        self.review_level += 1
        if not self.is_new_flashcard:
            self.times_correct += 1
        self.time_of_last_review = time.time()

    def decrement_review_level(self) -> None:
        """Decrement the review level but not below 1"""
        self.review_level = max(1, self.review_level - 1)
        self.time_of_last_review = time.time()

    def update_success_rate(self) -> None:
        """Update the success rate if not a new flashcard"""
        if not self.is_new_flashcard:
            self.reviews += 1
            self.success_rate = self.times_correct / self.reviews


class FlashcardOptimizer:
    def __init__(self):
        self.min_reviews_for_optimization = 5  # Minimum reviews needed before optimizing
        self.learning_rate = 0.1  # How aggressively to adjust parameters

    def optimize(self, scheduler) -> Tuple[float, float, int]:
        """
        Optimize parameters based on actual review performance.

        Args:
            scheduler: FlashcardScheduler with review history

        Returns:
            Tuple[float, float, int]: (initial_seconds_until_review, user_decay_constant, flashcard_limit)
        """
        if scheduler.total_flashcard_reviews < self.min_reviews_for_optimization:
            return (
                scheduler.initial_seconds_until_review,
                scheduler.user_decay_constant,
                scheduler.flashcard_limit
            )

        # Get all cards that have been reviewed
        reviewed_cards = [card for card in scheduler.review_flashcards + scheduler.due_flashcards
                        if not card.is_new_flashcard]

        if not reviewed_cards:
            return (
                scheduler.initial_seconds_until_review,
                scheduler.user_decay_constant,
                scheduler.flashcard_limit
            )

        # Analyze success rates at different intervals
        success_by_interval = defaultdict(list)
        for card in reviewed_cards:
            interval = scheduler.compute_secs_until_review(card.review_level, card)
            success_by_interval[interval].append(card.success_rate)

        # Optimize initial review time based on first review performance
        if scheduler.first_review_total > 0:
            first_review_rate = scheduler.first_review_successes / scheduler.first_review_total
            rate_difference = first_review_rate - scheduler.target_success_rate
            new_initial = scheduler.initial_seconds_until_review * (1.0 + rate_difference)
            new_initial = max(20.0, min(300.0, new_initial))
        else:
            new_initial = scheduler.initial_seconds_until_review

        # Optimize user decay constant based on overall performance
        avg_success_rate = scheduler.total_success_rate
        rate_difference = avg_success_rate - scheduler.target_success_rate
        new_user_decay = scheduler.user_decay_constant * (1.0 + rate_difference)
        new_user_decay = max(1.1, min(3.0, new_user_decay))

        print(f"\nOptimizing based on performance:")
        print(f"Average success rate: {avg_success_rate:.2%}")
        print(f"First review success rate: {first_review_rate:.2%}" if scheduler.first_review_total > 0 else "No first reviews yet")
        print(f"Adjusting initial review time: {scheduler.initial_seconds_until_review:.1f}s -> {new_initial:.1f}s")
        print(f"Adjusting user decay constant: {scheduler.user_decay_constant:.2f} -> {new_user_decay:.2f}")

        return (new_initial, new_user_decay, scheduler.flashcard_limit)


class FlashcardScheduler:
    def __init__(
            self,
            initial_seconds_until_review: float = 49.2,
            flashcard_limit: int = 8,
            target_success_rate: float = 0.8413,
            default_decay_constant: float = 1.5
    ):
        self.due_flashcards: Deque[Flashcard] = deque()
        self.review_flashcards: Deque[Flashcard] = deque()
        self.pending_flashcards: Deque[Flashcard] = deque()
        self.initial_seconds_until_review = initial_seconds_until_review
        self.flashcard_limit = flashcard_limit
        self.target_success_rate = target_success_rate

        # User's personal learning rate
        self.user_decay_constant: float = default_decay_constant

        # Dictionary mapping card questions to their decay constants
        self.card_decay_constants: Dict[str, float] = {}

        # Dictionary mapping card questions to their total reviews and performance sums
        self.card_stats: Dict[str, Dict[str, float]] = {}

        # Track first review performance
        self.first_review_successes: int = 0
        self.first_review_total: int = 0

        self.total_times_correct = 0
        self.total_flashcard_reviews = 0
        self.exit_flag = False

        self.default_deck_decay = 1.5  # Average decay for calibration
        self.deck_calibrations = {}  # Deck name -> average decay

    def adjust_initial_review_time(self) -> None:
        """Adjust initial review time based on first review performance"""
        if self.first_review_total > 0:
            first_review_success_rate = self.first_review_successes / self.first_review_total
            rate_difference = first_review_success_rate - self.target_success_rate
            adjustment_factor = 1.0 + rate_difference
            self.initial_seconds_until_review *= adjustment_factor
            self.initial_seconds_until_review = max(20.0, min(300.0, self.initial_seconds_until_review))

    def get_card_decay_constant(self, card: Flashcard) -> float:
        """Get card's personalized decay constant or initialize it."""
        if card.question not in self.card_decay_constants:
            # In practice, you'd look up the average card decay from a database
            # For now, using a default value
            self.initialize_card_decay(card, 1.5, "default_deck")

            self.card_stats[card.question] = {
                'total_reviews': 0,
                'sum_performance': 0
            }
        return self.card_decay_constants[card.question]

    def update_card_decay_constant(self, card: Flashcard, is_correct: bool) -> None:
        """Update card's personal decay constant based on performance."""
        if card.question not in self.card_stats:
            self.card_stats[card.question] = {
                'total_reviews': 0,
                'sum_performance': 0
            }

        stats = self.card_stats[card.question]
        stats['total_reviews'] += 1
        performance_score = 1.0 if is_correct else 0.0
        stats['sum_performance'] += performance_score

        avg_performance = stats['sum_performance'] / stats['total_reviews']
        rate_difference = avg_performance - self.target_success_rate
        adjustment_factor = 1.0 + rate_difference

        old_decay = self.card_decay_constants[card.question]
        new_decay = old_decay * adjustment_factor
        new_decay = max(1.1, min(3.0, new_decay))
        self.card_decay_constants[card.question] = new_decay

        print(f"\nUpdating card decay for '{card.question}':")
        print(f"- Performance rate: {avg_performance:.3f}")
        print(f"- Old decay: {old_decay:.3f}")
        print(f"- New decay: {new_decay:.3f}")

    def adjust_user_decay_constant(self, success: bool, card: Flashcard) -> None:
        """Adjust user decay constant based only on non-first reviews"""
        if not card.is_new_flashcard:  # Only adjust for subsequent reviews
            card_decay = self.get_card_decay_constant(card)
            rate_difference = (1.0 if success else 0.0) - self.target_success_rate
            difficulty_factor = card_decay / 1.5
            adjustment = rate_difference * difficulty_factor

            self.user_decay_constant *= (1.0 + adjustment)
            self.user_decay_constant = max(1.1, min(3.0, self.user_decay_constant))

    def calculate_current_efficiency(self) -> float:
        """Calculate current learning efficiency."""
        if not (self.due_flashcards or self.review_flashcards):
            return 0.0

        total_cards = len(self.due_flashcards) + len(self.review_flashcards)
        all_cards = list(self.due_flashcards) + list(self.review_flashcards)

        efficiency_sum = 0
        current_time = time.time()

        for card in all_cards:
            time_invested = current_time - card.time_of_last_review
            if time_invested > 0 and card.reviews > 0:
                efficiency = card.success_rate / (time_invested / 3600)  # per hour
                efficiency_sum += efficiency

        return efficiency_sum / total_cards if total_cards > 0 else 0.0

    def calculate_retention_score(self) -> float:
        """Calculate long-term retention score."""
        all_cards = list(self.due_flashcards) + list(self.review_flashcards)
        if not all_cards:
            return 0.0

        retention_sum = 0
        for card in all_cards:
            if card.reviews > 0:
                retention_sum += card.success_rate * (1 + 0.1 * card.review_level)

        return retention_sum / len(all_cards)

    def calculate_time_management_score(self) -> float:
        """Calculate how well we're staying within flashcard limits."""
        total_cards = len(self.due_flashcards) + len(self.review_flashcards)
        if total_cards == 0:
            return 1.0
        if total_cards <= self.flashcard_limit:
            return 1.0
        return max(0.0, 1.0 - (total_cards - self.flashcard_limit) / self.flashcard_limit)

    def run_optimization_cycle(self) -> Tuple[float, float, int]:
        """Run an optimization cycle and update parameters."""
        optimizer = FlashcardOptimizer()
        new_initial, new_user_decay, new_limit = optimizer.optimize(self)

        # Actually apply the optimized parameters
        self.initial_seconds_until_review = new_initial
        self.user_decay_constant = new_user_decay
        self.flashcard_limit = new_limit

        return (new_initial, new_user_decay, new_limit)

    @property
    def total_success_rate(self) -> float:
        if self.total_flashcard_reviews == 0:
            return 0.0
        return self.total_times_correct / self.total_flashcard_reviews

    def compute_secs_until_review(self, current_review_level: int, card: Flashcard) -> float:
        """Compute review interval using only the personalized card decay constant."""
        if current_review_level <= 1:
            return self.initial_seconds_until_review

        card_decay = self.get_card_decay_constant(card)
        interval = self.initial_seconds_until_review * (card_decay ** current_review_level)

        return round(min(interval, 3600.0), 2)

    def save_state(self, filepath: Path) -> None:
        try:
            saved_time = datetime.now().timestamp()
            flashcard_data = {
                'due_flashcards': self.due_flashcards,
                'review_flashcards': self.review_flashcards,
                'pending_flashcards': self.pending_flashcards,
                'saved_time': saved_time,
                'params': {
                    'initial_seconds_until_review': self.initial_seconds_until_review,
                    'flashcard_limit': self.flashcard_limit,
                    'user_decay_constant': self.user_decay_constant,
                    'card_decay_constants': self.card_decay_constants,
                    'card_stats': self.card_stats,
                    'first_review_successes': self.first_review_successes,
                    'first_review_total': self.first_review_total
                }
            }

            with open(filepath, 'wb') as file:
                pickle.dump(flashcard_data, file)
        except IOError as e:
            raise IOError(f"Failed to save flashcard data: {str(e)}")

    def load_state(self, filepath: Path) -> None:
        try:
            with open(filepath, 'rb') as file:
                data = pickle.load(file)

            if not all(key in data for key in
                       ['due_flashcards', 'review_flashcards', 'pending_flashcards', 'saved_time',
                        'params']):
                raise ValueError("Saved data is missing required fields")

            elapsed_time = datetime.now().timestamp() - data['saved_time']
            self._update_review_times(elapsed_time)

            self.due_flashcards = data['due_flashcards']
            self.review_flashcards = data['review_flashcards']
            self.pending_flashcards = data['pending_flashcards']
            self.initial_seconds_until_review = data['params']['initial_seconds_until_review']
            self.flashcard_limit = data['params']['flashcard_limit']
            self.user_decay_constant = data['params']['user_decay_constant']
            self.card_decay_constants = data['params']['card_decay_constants']
            self.card_stats = data['params']['card_stats']
            self.first_review_successes = data['params']['first_review_successes']
            self.first_review_total = data['params']['first_review_total']

        except (FileNotFoundError, pickle.UnpicklingError) as e:
            raise ValueError(f"Failed to load flashcard data: {str(e)}")

    def _update_review_times(self, elapsed_time: float) -> None:
        """Updates all flashcard review times based on elapsed time."""
        sorted_due_flashcards = []

        for card in list(self.review_flashcards):
            card.seconds_until_review -= round(elapsed_time, 2)
            if card.seconds_until_review <= 0:
                card.seconds_until_review = 0
                sorted_due_flashcards.append(card)

        sorted_due_flashcards.sort(key=lambda c: c.seconds_until_review)
        for card in sorted_due_flashcards:
            self.due_flashcards.append(card)
            self.review_flashcards.remove(card)

    def add_flashcard(self, question: str, answer: str) -> None:
        """Adds a new flashcard to the pending queue."""
        new_card = Flashcard(question=question, answer=answer)
        self.pending_flashcards.append(new_card)

    def print_all_cards_status(self):
        print("\nAll cards status:")
        all_cards = list(self.review_flashcards) + list(self.due_flashcards)
        current_time = time.time()
        for card in all_cards:
            if card in self.review_flashcards:
                elapsed = current_time - card.time_of_last_review
                remaining = max(0, card.seconds_until_review - elapsed)
                print(f"'{card.question}': {remaining:.1f} seconds remaining")
            else:
                print(f"'{card.question}': due now")

    def process_answer(self, card: Flashcard, is_correct: bool) -> None:
        """Process answer and update both user and card decay constants"""
        if card.is_new_flashcard:
            self.first_review_total += 1
            if is_correct:
                self.first_review_successes += 1
            self.adjust_initial_review_time()

        if is_correct:
            card.increment_review_level()
            self.total_times_correct += 1
            if not card.is_new_flashcard:
                self.adjust_user_decay_constant(True, card)
                self.update_card_decay_constant(card, True)
        else:
            card.decrement_review_level()
            if not card.is_new_flashcard:
                self.adjust_user_decay_constant(False, card)
                self.update_card_decay_constant(card, False)

        card.update_success_rate()
        self.total_flashcard_reviews += 1

        card.seconds_until_review = self.compute_secs_until_review(
            card.review_level,
            card
        )

        card.is_new_flashcard = False

        print(f"\nCard: '{card.question}'")
        print(f"Level: {card.review_level}")
        print(f"Next review in: {card.seconds_until_review:.1f} seconds")

        card.time_of_last_review = time.time()
        self.review_flashcards.append(card)

        self.print_all_cards_status()

    def update_due_cards(self) -> None:
        current_time = time.time()
        next_due_time = float('inf')
        next_card_question = None
        for card in self.review_flashcards:
            elapsed = current_time - card.time_of_last_review
            time_remaining = card.seconds_until_review - elapsed
            if time_remaining < next_due_time:
                next_due_time = time_remaining
                next_card_question = card.question

        if next_due_time < float('inf'):
            if next_due_time > 0:
                print(f"\nWaiting {next_due_time:.1f} seconds until next card ('{next_card_question}')...")
                time.sleep(next_due_time)

            current_time = time.time()
            for card in list(self.review_flashcards):
                elapsed = current_time - card.time_of_last_review
                if elapsed >= card.seconds_until_review:
                    self.due_flashcards.append(card)
                    self.review_flashcards.remove(card)

        def add_deck_calibration(self, deck_name: str, avg_deck_decay: float = 1.5) -> None:
            """Add or update average decay constant for a deck."""
            self.deck_calibrations[deck_name] = avg_deck_decay

        def get_relative_user_performance(self) -> float:
            """Calculate user's relative performance compared to average."""
            return self.user_decay_constant / self.default_deck_decay

        def initialize_card_decay(self, card: Flashcard, avg_card_decay: float, deck_name: str) -> None:
            """Initialize a card's decay constant based on user's relative performance."""
            if deck_name not in self.deck_calibrations:
                self.deck_calibrations[deck_name] = self.default_deck_decay


class FlashcardApp(ctk.CTk):
    def __init__(self, scheduler):
        super().__init__()

        self.scheduler = scheduler
        self.current_card = None

        # Configure window
        self.title("Flashcards")
        self.geometry("400x240")
        self.resizable(False, False)

        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=3)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Card display label
        self.card_label = ctk.CTkLabel(
            self.main_frame,
            text="Ready to start!",
            font=("Arial", 20),
            wraplength=350,
            justify="center"
        )
        self.card_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")

        # Button frame
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)

        # Control buttons
        self.show_answer_btn = ctk.CTkButton(
            self.button_frame,
            text="Show Answer",
            command=self.show_answer,
            font=("Arial", 16)
        )
        self.show_answer_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Response buttons frame
        self.response_frame = ctk.CTkFrame(self.button_frame)
        self.response_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.response_frame.grid_rowconfigure(0, weight=1)
        self.response_frame.grid_rowconfigure(1, weight=1)
        self.response_frame.grid_columnconfigure(0, weight=1)

        self.correct_btn = ctk.CTkButton(
            self.response_frame,
            text="Correct",
            command=lambda: self.process_response(True),
            font=("Arial", 16),
            fg_color="green"
        )
        self.correct_btn.grid(row=0, column=0, padx=5, pady=2, sticky="ew")

        self.incorrect_btn = ctk.CTkButton(
            self.response_frame,
            text="Incorrect",
            command=lambda: self.process_response(False),
            font=("Arial", 16),
            fg_color="red"
        )
        self.incorrect_btn.grid(row=1, column=0, padx=5, pady=2, sticky="ew")

        # Initial button states
        self.correct_btn.configure(state="disabled")
        self.incorrect_btn.configure(state="disabled")

        # Bind keyboard shortcuts
        self.bind('<space>', lambda e: self.show_answer())
        self.bind('<Return>', lambda e: self.show_answer())
        self.bind('<Up>', lambda e: self.process_response(True))
        self.bind('<Down>', lambda e: self.process_response(False))

        # Start update cycle
        self.update_cards()

    def show_answer(self):
        if self.current_card and self.show_answer_btn.cget('state') == 'normal':
            # Print debug info
            print("\nAnswer:", self.current_card.answer)

            self.card_label.configure(text=f"Answer:\n{self.current_card.answer}")
            self.show_answer_btn.configure(state="disabled")
            self.correct_btn.configure(state="normal")
            self.incorrect_btn.configure(state="normal")
            # Remove the card from due_flashcards once we show the answer
            if self.current_card in self.scheduler.due_flashcards:
                self.scheduler.due_flashcards.remove(self.current_card)

    def process_response(self, is_correct):
        if self.current_card:
            # Print debug information first
            print(f"\nCard: '{self.current_card.question}'")

            # Process the answer
            self.scheduler.process_answer(self.current_card, is_correct)

            # Print more debug info
            print(f"Level: {self.current_card.review_level}")
            print(f"Next review in: {self.current_card.seconds_until_review:.1f} seconds")
            print("\nAll cards status:")
            for card in self.scheduler.review_flashcards:
                elapsed = time.time() - card.time_of_last_review
                remaining = max(0, card.seconds_until_review - elapsed)
                print(f"'{card.question}': {remaining:.1f} seconds remaining")
            for card in self.scheduler.due_flashcards:
                print(f"'{card.question}': due now")

            # Update UI state
            self.show_answer_btn.configure(state="normal")
            self.correct_btn.configure(state="disabled")
            self.incorrect_btn.configure(state="disabled")

            # Clear current card and update
            self.current_card = None
            self.update_cards()

    def update_cards(self):
        """Only called when we know a card is due"""
        current_time = time.time()

        # Move pending cards if needed
        while (len(self.scheduler.due_flashcards) + len(self.scheduler.review_flashcards)
               < self.scheduler.flashcard_limit and self.scheduler.pending_flashcards):
            self.scheduler.due_flashcards.append(self.scheduler.pending_flashcards.popleft())

        # Find if any cards are due
        for card in list(self.scheduler.review_flashcards):
            elapsed = current_time - card.time_of_last_review
            if elapsed >= card.seconds_until_review:
                self.scheduler.review_flashcards.remove(card)
                self.scheduler.due_flashcards.append(card)
                print(f"Card '{card.question}' is now due")

        # Show card if needed
        if self.scheduler.due_flashcards and (
                not self.current_card or
                (self.show_answer_btn.cget('state') == 'disabled' and
                 self.correct_btn.cget('state') == 'disabled')
        ):
            self.current_card = self.scheduler.due_flashcards[0]
            print("\nQuestion:", self.current_card.question)
            self.card_label.configure(text=f"Question:\n{self.current_card.question}")
            self.show_answer_btn.configure(state="normal")
            self.correct_btn.configure(state="disabled")
            self.incorrect_btn.configure(state="disabled")
        elif not self.scheduler.due_flashcards and not self.current_card:
            self.card_label.configure(text="No cards due for review")
            self.show_answer_btn.configure(state="disabled")

            # Schedule next update for when the next card will be due
            if self.scheduler.review_flashcards:
                next_due_time = float('inf')
                next_card = None
                for card in self.scheduler.review_flashcards:
                    time_until_due = (card.time_of_last_review + card.seconds_until_review) - current_time
                    if time_until_due < next_due_time:
                        next_due_time = time_until_due
                        next_card = card

                if next_card:
                    print(f"\nWaiting {next_due_time:.1f} seconds until next card ('{next_card.question}')...")
                    # Schedule exactly when the next card is due
                    self.after(int(next_due_time * 1000), self.update_cards)

    def on_closing(self):
        """Handle window closing"""
        try:
            self.scheduler.save_state(Path("flashcards_data.pkl"))
            print("\nSaving state and exiting...")
        except Exception as e:
            print(f"Error saving state: {e}")
        self.quit()


def main():
    # Create scheduler and try to load existing state
    scheduler = FlashcardScheduler()

    try:
        scheduler.load_state(Path("flashcards_data.pkl"))
        print("Loaded existing flashcard state")
    except Exception as e:
        print(f"Starting with new flashcard state: {e}")
        # Add sample flashcards only if no state was loaded
        scheduler.add_flashcard("Empty save file sample: What is your name?", "Joseph")
        scheduler.add_flashcard("Empty save file sample: What is your nickname?", "JoJo")

    # Debug prints
    print("\nDebug - Checking scheduler state:")
    print(f"Pending flashcards: {len(scheduler.pending_flashcards)}")
    print(f"Due flashcards: {len(scheduler.due_flashcards)}")
    print(f"Review flashcards: {len(scheduler.review_flashcards)}")

    if scheduler.pending_flashcards:
        print("\nPending cards:")
        for card in scheduler.pending_flashcards:
            print(f"- {card.question}")
    if scheduler.due_flashcards:
        print("\nDue cards:")
        for card in scheduler.due_flashcards:
            print(f"- {card.question}")
    if scheduler.review_flashcards:
        print("\nReview cards:")
        for card in scheduler.review_flashcards:
            print(f"- {card.question}")

    # Create and run app
    app = FlashcardApp(scheduler)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()