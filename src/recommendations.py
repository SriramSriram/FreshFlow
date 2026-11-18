
import numpy as np


def classify_expiry_risk(days_remaining):
    """
    Classify expiry risk based on remaining shelf life.
    Thresholds are provisional.
    """

    if days_remaining <= 3:
        return "Critical"
    elif days_remaining <= 7:
        return "High"
    elif days_remaining <= 14:
        return "Moderate"
    else:
        return "Low"


def combined_inventory_recommendation(
    predicted_demand,
    available_stock,
    days_remaining
):
    """
    Generate an inventory recommendation using
    predicted demand, available stock and expiry risk.
    """

    shortage = max(0, predicted_demand - available_stock)

    expiry_risk = classify_expiry_risk(days_remaining)

    if shortage > 0 and expiry_risk in ["Critical", "High"]:
        status = "Review Replenishment + Expiry"
        action = "Check existing stock before ordering"

    elif shortage > 0:
        status = "Reorder Required"
        action = "Consider replenishing the shortage"

    elif expiry_risk in ["Critical", "High"]:
        status = "Expiry Risk"
        action = "Prioritize existing stock for sale"

    else:
        status = "Stock Sufficient"
        action = "Continue monitoring inventory"

    return {
        "Predicted Demand": round(float(predicted_demand), 2),
        "Available Stock": int(available_stock),
        "Potential Shortage": round(float(shortage), 2),
        "Potential Reorder Quantity": int(np.ceil(shortage)),
        "Days Remaining": int(days_remaining),
        "Expiry Risk": expiry_risk,
        "Final Status": status,
        "Recommended Action": action
    }
