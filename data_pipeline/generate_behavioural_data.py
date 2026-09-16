"""
Synthtic behavioural data generator for phisihing susceptibility.

Simulates per-user interaction features (hover time, click speed, etc.)
calibrated to findings from the behavioural phishing literature, since
no public dataset with these features exists (see Methodology writeup).
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_USERS = 2000 #number of simulated user to generate

#Generating hidden susceptibility profile scores
def generate_profile_scores(n_users: int)-> np.ndarray:
    """
    Each simulated data gets hidden 'susceptibility profile' score 
    between 0 (cautions) and 1(susceptible).
    We use a Beta distribution because it's bounded to [0, 1] (unlike a
    normal distribution, which could give nonsensical negative scores)
    and its two shape parameters let us control the population's overall
    skew. 
    """
    return np.random.beta(a=2, b=3, size=n_users)

#Feature 1: Time-to-First-Click (or non-click decision)
def generate_time_to_first_click(profile_scores: np.ndarray)-> np.ndarray:
    """
    Seconds from opening the email to clicking the link (or deciding
    not to). Grounded in Lain et al.'s finding that susceptible users
    act fast, without pausing to evaluate.
    using lognormal distrubution because reaction times cannot be 
    negative and are typically right-skewed real data.
    """
    median_seconds = 35 - 30 * profile_scores
    mu = np.log(median_seconds)
    sigma = 0.5

    times = np.random.lognormal(mean=mu, sigma=sigma)
    return np.clip(times, 1, 120) #clip to plausible 1-120 second range

# Feature 2: Hover/dwell time on the link
def generate_hover_time(profile_scores:np.ndarray) -> np.ndarray:
    """
    Seconds the cursor rests on the link before a decision is made.
    Here Low profile score (cautations) means Longer hover.Cautious 
    users hover to check where the link really points; 
    susceptible users barely pause before clicking.
    """
    median_hover = 0.3 + 2.7 * (1 - profile_scores)
    mu = np.log(median_hover)
    sigma = 0.6

    hover_times = np.random.lognormal(mean=mu, sigma=sigma)
    return np.clip(times, 1, 120) #clip to plausible 1-120 second range

# Feature 3: Number of link hovers before deciding 
def generate_hover_count(profile_scores: np.ndarray)->np.ndarray:
    """
    How many times the user hovers over the link (or other elements)
    before acting - repeated hovering signals active comparison/
    verification behaviour, not just a single pass.
 
    We use a Poisson distribution here instead of lognormal, because
    this is a COUNT of discrete events (0, 1, 2, 3...), not a
    continuous time value. Poisson is the standard distribution for
    'how many times did X happen' data.
    """
    lambda_hovers=0.4 + 2.6 * (1 - profile_scores)
    return np.random.poisson(lam = lambda_hovers)

# Feature 4: Mean mouse moment speed
def generate_mouse_speed(profile_scores: np.ndarray)->np.ndarray:
    """
    Average cursor speed (pixels/second) across the session.
    """
    median_speed = 200 + 600 * profile_scores #pixels/second
    mu = np.log(median_speed)
    sigma=0.4

    speeds = np.random.lognormal(mean=mu, sigma=sigma)
    return np.clip(speeds, 50, 2000) #clip to a plausible 50-2000 px/s range

#Feature 5 - slow-moment ratio (strongest single predictor)
def generate_slow_movement_ratio(profile_scores: np.ndarray)->np.ndarray:
    """
    Proportion of the session spent in 'slow' mouse movement (vs fast).
    Flagged in the Springer mouse-behaviour paper as one of the
    strongest single predictors of phishing awareness - so we make
    this feature's relationship to the profile score TIGHTER (less
    overlap between cautious and susceptible users) than the others.

    """
    target_mean = 0.75 - 0.65 * profile_scores # caution -0.75, susceptile -0.10
    concentration = 15

    alpha = target_mean * concentration
    beta = (1 - target_mean)*concentration

    return np.random.beta(alpha, beta)

#Feature 6: Email reopen count
def generate_reopen_count(profile_scores: np.ndarray)->np.ndarray:
    """
    How many times the user reopens/reviews the email before acting.
    Reopening signals doubt or a second look forming - cautious users
    reopen to compare against other sources; susceptible users act on
    the first read.
    """
    lambda_reopen = 0.1 * 1.2 + (1-profile_scores)
    return np.random.poision(lam=lambda_reopen)

#Feature 7: Sender address check - Binary(checked/ didn't checked)
def generate_sender_check(profile_scores: np.ndarry)-> np.ndarray:
    """
    Whether the user's focus/hover ever goes to the 'From' field -
    one of the strongest manual verification behaviours cited across
    the phishing literature.
    """
    p_check = 0.85 - 0.75 * profile_scores
    p_check = np.clip(p_check, 0.02, 0.98) #keep away from hard 0/1 probabilities

    return np.random.binomial(n=1, p=p_check)
# click/no-click label
def generate_click_label(profile_scores: np.ndarray)->np.ndarray:
    """
    The ground-truth outcome: did this user click the phishing link?
 
    Crucially, this is PROBABILISTIC, not a hard cutoff on profile
    score. A profile score of 0.8 means an 80% chance of clicking, not
    a guarantee - this produces realistic overlap between the two
    classes (some cautious-leaning users still slip up, some
    susceptible-leaning users happen to catch it this time), which is
    what makes this an actual classification problem for the MLP
    rather than a trivial lookup table.
    """
    click_probability= profile_scores #profile scores IS the Click probability
    return np.random.bionomial(n=1, p=click_probability)

#Feature 8: Time to report (only meaningful for non clickers)
def generate_time_to_report(profile_scores: np.ndarray, click_label:np.ndarray)->np.ndarray:
    """
    For users who did NOT click, how long before they report/flag the
    email. Captures the 'positive' defensive outcome, not just
    avoidance - relevant to the dashboard's real-world framing.
 
    This feature is only defined for non-clickers. For users who
    clicked, we set it to NaN (missing) rather than 0 or some dummy
    value - 0 would falsely imply 'reported instantly', and this gives
    our cleaning pipeline (the next stage) a real, structural missing-
    data pattern to handle, rather than only the randomly injected
    noise we'll add later.
 
    Among non-clickers, cautious users (low profile score) report
    faster - they're actively engaged in defending, not just passively
    avoiding.
    """
    median_report_seconds = 30 + 120 * profile_scores #cautions -30s, less cautions -150s
    mu = np.log(median_report_seconds)
    sigma=0.6

    report_times = np.random.lognormal(mean=mu, sigma=sigma)
    report_times= np.clip(report_times, 5, 600)

    #only non-clickers have a report time; clickers get N/A
    report_times=np.where(click_label==0, report_times, np.nan)
    return report_times

if __name__=="__main__":
    profile_scores = generate_profile_scores(N_USERS)
    time_to_click = generate_time_to_first_click(profile_scores)
    hover_time = generate_hover_time(profile_scores)
    hover_count = generate_hover_count(profile_scores)
    mouse_speed = generate_mouse_speed(profile_scores)
    slow_ratio = generate_slow_movement_ratio(profile_scores)
    reopen_count = generate_reopen_count(profile_scores)
    sender_check = generate_sender_check(profile_scores)
    click_label = generate_click_label(profile_scores)
    time_to_report = generate_time_to_report(profile_scores, click_label)
 
    print(f"Generated {N_USERS} profile scores")
    print(f"Profile mean: {profile_scores.mean():.3f}")
    print()
 
    cautious_mask = profile_scores < 0.3
    susceptible_mask = profile_scores > 0.7
 
    print("Time-to-first-click sanity check:")
    print(f"  Cautious users mean: {time_to_click[cautious_mask].mean():.1f}s")
    print(f"  Susceptible users mean: {time_to_click[susceptible_mask].mean():.1f}s")
    print()
    print("Hover time sanity check:")
    print(f"  Cautious users mean: {hover_time[cautious_mask].mean():.2f}s")
    print(f"  Susceptible users mean: {hover_time[susceptible_mask].mean():.2f}s")
    print()
    print("Hover count sanity check:")
    print(f"  Cautious users mean: {hover_count[cautious_mask].mean():.2f} hovers")
    print(f"  Susceptible users mean: {hover_count[susceptible_mask].mean():.2f} hovers")
    print()
    print("Mouse speed sanity check:")
    print(f"  Cautious users mean: {mouse_speed[cautious_mask].mean():.0f} px/s")
    print(f"  Susceptible users mean: {mouse_speed[susceptible_mask].mean():.0f} px/s")
    print()
    print("Slow-movement ratio sanity check:")
    print(f"  Cautious users mean: {slow_ratio[cautious_mask].mean():.2f}")
    print(f"  Susceptible users mean: {slow_ratio[susceptible_mask].mean():.2f}")
    print()
    print("Reopen count sanity check:")
    print(f"  Cautious users mean: {reopen_count[cautious_mask].mean():.2f} reopens")
    print(f"  Susceptible users mean: {reopen_count[susceptible_mask].mean():.2f} reopens")
    print()
    print("Sender-check sanity check:")
    print(f"  Cautious users: {sender_check[cautious_mask].mean()*100:.0f}% checked sender")
    print(f"  Susceptible users: {sender_check[susceptible_mask].mean()*100:.0f}% checked sender")
    print()
    print("Click label sanity check:")
    print(f"  Overall click rate: {click_label.mean()*100:.1f}%")
    print(f"  Cautious users click rate: {click_label[cautious_mask].mean()*100:.1f}%")
    print(f"  Susceptible users click rate: {click_label[susceptible_mask].mean()*100:.1f}%")
    print()
    print("Time-to-report sanity check:")
    n_reported = (~np.isnan(time_to_report)).sum()
    print(f"  Non-clickers with a report time: {n_reported} out of {(click_label == 0).sum()} non-clickers")
    print(f"  Mean report time among non-clickers: {np.nanmean(time_to_report):.1f}s")
 
