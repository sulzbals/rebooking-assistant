"""The rebooking assistant as a LangGraph graph: find the fare rules, check seats, write the answer."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from assistant import booking, llm
from assistant.data import PASSAGES
from assistant.retrieval import CHOSEN_RETRIEVER, RETRIEVERS


class State(TypedDict, total=False):
    question: str
    flight: str
    passages: list
    sources: list
    seats: int | None
    booking_failed: bool
    attempts: int
    answer: str


def find_rules(state):
    passages = RETRIEVERS[CHOSEN_RETRIEVER](state["question"], PASSAGES, 3)
    return {"passages": passages, "sources": [p["id"] for p in passages]}


MAX_BOOKING_ATTEMPTS = 3


def check_seats(state):
    attempts = state.get("attempts", 0) + 1
    try:
        return {"seats": booking.free_seats(state["flight"]), "booking_failed": False, "attempts": attempts}
    except booking.BookingSystemDown:
        return {"seats": None, "booking_failed": True, "attempts": attempts}


def after_check_seats(state):
    if state["booking_failed"] and state["attempts"] < MAX_BOOKING_ATTEMPTS:
        return "check_seats"
    return "write_answer"


def write_answer(state):
    return {"answer": llm.write_answer(state["question"], state["passages"], state["seats"])}


def build():
    g = StateGraph(State)
    g.add_node("find_rules", find_rules)
    g.add_node("check_seats", check_seats)
    g.add_node("write_answer", write_answer)
    g.add_edge(START, "find_rules")
    g.add_edge("find_rules", "check_seats")
    g.add_conditional_edges("check_seats", after_check_seats, ["check_seats", "write_answer"])
    g.add_edge("write_answer", END)
    return g.compile()


def ask(question, flight):
    return build().invoke({"question": question, "flight": flight})
