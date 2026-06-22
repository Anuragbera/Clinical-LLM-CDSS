# =========================
# ANEMIA SEVERITY (SIMPLE RULE)
# =========================
def anemia_severity(labs, predicted_labels):

    num_types = len(predicted_labels)

    if num_types <= 1:
        severity = "Normal"
    elif num_types == 2:
        severity = "Mild"
    elif num_types == 3:
        severity = "Moderate"
    else:
        severity = "Severe"

    advice_map = {
        "Normal": "No significant anemia detected.",
        "Mild": "Monitor diet and consider supplements.",
        "Moderate": "Consult a doctor and start treatment.",
        "Severe": "Immediate medical attention required."
    }

    return severity, advice_map[severity]


# =========================
# LIVER SEVERITY (FROM MODEL OUTPUT)
# =========================
def liver_severity(prediction):

    # prediction = "Class A" / "Class B" / "Class C"

    if "A" in prediction:
        severity = "Mild"
    elif "B" in prediction:
        severity = "Moderate"
    else:
        severity = "Severe"

    advice_map = {
        "Mild": "Liver condition is stable. Routine monitoring advised.",
        "Moderate": "Liver dysfunction present. Medical treatment required.",
        "Severe": "Advanced liver disease. Urgent specialist care needed."
    }

    return severity, advice_map[severity]