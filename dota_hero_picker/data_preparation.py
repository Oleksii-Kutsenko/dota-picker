import logging
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import settings

from .load_personal_matches import get_hero_data

logger = logging.getLogger(__name__)
synergy = pd.read_csv(settings.MATCHUPS_STATISTICS_PATH / Path("synergy.csv"))
counters = pd.read_csv(
    settings.MATCHUPS_STATISTICS_PATH / Path("counters.csv")
)

hero_data = get_hero_data()
heroes = [hero_data_item["localized_name"] for hero_data_item in hero_data]
num_heroes = len(heroes)
hero_to_id = {hero: idx for idx, hero in enumerate(heroes)}
id_to_hero = {idx: hero for hero, idx in hero_to_id.items()}
embedding_dim = 32


def create_pair_mappings_and_init(
    num_heroes, embedding_dim, synergy_df, counters_df
):
    synergy_pair_to_idx = {}
    synergy_init = [torch.zeros(embedding_dim)]
    idx = 1
    for i in range(num_heroes):
        for j in range(i + 1, num_heroes):
            synergy_pair_to_idx[(i, j)] = idx
            s_score = synergy_df[
                (synergy_df["hero_id_1"] == i) & (synergy_df["hero_id_2"] == j)
            ]["score"]
            score = s_score.iloc[0] if not s_score.empty else 0.0
            init_vec = torch.randn(embedding_dim) * 0.01
            init_vec[0] = score
            synergy_init.append(init_vec)
            idx += 1

    counter_pair_to_idx = {}
    counter_init = [torch.zeros(embedding_dim)]
    idx = 1
    for i in range(num_heroes):
        for j in range(num_heroes):
            if i != j:
                counter_pair_to_idx[(i, j)] = idx
                c_score = counters_df[
                    (counters_df["hero_id_1"] == i)
                    & (counters_df["hero_id_2"] == j)
                ]["score"]
                score = c_score.iloc[0] if not c_score.empty else 0.0
                init_vec = torch.randn(embedding_dim) * 0.01
                init_vec[0] = score
                counter_init.append(init_vec)
                idx += 1

    return (
        synergy_pair_to_idx,
        torch.stack(synergy_init),
        counter_pair_to_idx,
        torch.stack(counter_init),
    )


(
    synergy_pair_to_idx,
    synergy_init_table,
    counter_pair_to_idx,
    counter_init_table,
) = create_pair_mappings_and_init(num_heroes, embedding_dim, synergy, counters)
num_synergy_pairs = len(synergy_pair_to_idx)
num_counter_pairs = len(counter_pair_to_idx)


def create_pairwise_indices(team_picks, opponent_picks):
    team_synergy_pairs = []
    for hero1, hero2 in combinations([hero for hero in team_picks if hero], 2):
        hero_id_1 = hero_to_id.get(hero1, -1)
        hero_id_2 = hero_to_id.get(hero2, -1)
        if hero_id_1 == -1 or hero_id_2 == -1:
            continue
        pair = tuple(sorted([hero_id_1, hero_id_2]))
        idx = synergy_pair_to_idx.get(pair)
        if idx is not None:
            team_synergy_pairs.append(idx)

    opponent_synergy_pairs = []
    for hero1, hero2 in combinations(
        [hero for hero in opponent_picks if hero], 2
    ):
        hero_id_1 = hero_to_id.get(hero1, -1)
        hero_id_2 = hero_to_id.get(hero2, -1)
        if hero_id_1 == -1 or hero_id_2 == -1:
            continue
        pair = tuple(sorted([hero_id_1, hero_id_2]))
        idx = synergy_pair_to_idx.get(pair)
        if idx is not None:
            opponent_synergy_pairs.append(idx)

    team_counter_pairs = []
    for team_hero in [hero for hero in team_picks if hero]:
        for opp_hero in [hero for hero in opponent_picks if hero]:
            team_hero_id = hero_to_id.get(team_hero, -1)
            opp_hero_id = hero_to_id.get(opp_hero, -1)
            if team_hero_id == -1 or opp_hero_id == -1:
                continue
            pair = (team_hero_id, opp_hero_id)
            idx = counter_pair_to_idx.get(pair)
            if idx is not None:
                team_counter_pairs.append(idx)

    return team_synergy_pairs, opponent_synergy_pairs, team_counter_pairs


def create_aggregate_features(team_picks, opponent_picks):
    features = []

    team_synergies = []
    for hero1, hero2 in combinations([hero for hero in team_picks if hero], 2):
        hero_id_1 = hero_to_id[hero1]
        hero_id_2 = hero_to_id[hero2]

        s_score = synergy[
            (synergy["hero_id_1"] == hero_id_1)
            & (synergy["hero_id_2"] == hero_id_2)
        ]["score"]
        if not s_score.empty:
            team_synergies.append(s_score.iloc[0])

    opponent_synergies = []
    for hero1, hero2 in combinations(
        [hero for hero in opponent_picks if hero], 2
    ):
        hero_id_1 = hero_to_id[hero1]
        hero_id_2 = hero_to_id[hero2]

        s_score = synergy[
            (synergy["hero_id_1"] == hero_id_1)
            & (synergy["hero_id_2"] == hero_id_2)
        ]["score"]
        if not s_score.empty:
            opponent_synergies.append(s_score.iloc[0])
    features.extend(
        [
            np.mean(team_synergies) if team_synergies else 0,
            np.std(team_synergies) if team_synergies else 0,
            np.mean(opponent_synergies) if opponent_synergies else 0,
            np.std(opponent_synergies) if opponent_synergies else 0,
        ]
    )

    team_counters = []
    enemy_counters = []

    for team_hero in [hero for hero in team_picks if hero]:
        for opp_hero in [hero for hero in opponent_picks if hero]:
            team_hero_id, opp_hero_id = (
                hero_to_id[team_hero],
                hero_to_id[opp_hero],
            )

            c_score = counters[
                (counters["hero_id_1"] == team_hero_id)
                & (counters["hero_id_2"] == opp_hero_id)
            ]["score"]
            if not c_score.empty:
                team_counters.append(c_score.iloc[0])
                enemy_counters.append(c_score.iloc[0] * -1)

    features.extend(
        [
            np.mean(team_counters) if team_counters else 0,
            np.std(team_counters) if team_counters else 0,
            np.mean(enemy_counters) if enemy_counters else 0,
            np.std(enemy_counters) if enemy_counters else 0,
            (np.mean(team_counters) - np.mean(enemy_counters))
            if team_counters and enemy_counters
            else 0,
            len([hero for hero in team_picks if hero]),
            len([hero for hero in opponent_picks if hero]),
        ]
    )
    return np.array(features)


def create_input_vector(
    team_picks: list, opponent_picks: list, actual_pick: str
) -> tuple:
    # Vector for your team's heroes (size = num_heroes)
    team_vec = np.zeros(num_heroes)
    for index, hero in enumerate(team_picks, start=1):
        if hero and hero in hero_to_id:
            team_vec[hero_to_id[hero]] = index

    # Vector for opponent's heroes (size = num_heroes)
    opponent_vec = np.zeros(num_heroes)
    for index, hero in enumerate(opponent_picks, start=1):
        if hero and hero in hero_to_id:
            opponent_vec[hero_to_id[hero]] = index

    # One-hot vector for the hero pick being evaluated
    pick_vec = np.zeros(num_heroes)
    if actual_pick in hero_to_id:
        pick_vec[hero_to_id[actual_pick]] = len(team_picks) + 1

    team_syn_pairs, opp_syn_pairs, team_cnt_pairs = create_pairwise_indices(
        team_picks, opponent_picks
    )
    aggregate_features = create_aggregate_features(team_picks, opponent_picks)
    return (
        team_vec,
        opponent_vec,
        pick_vec,
        team_syn_pairs,
        opp_syn_pairs,
        team_cnt_pairs,
        aggregate_features,
    )


def augment_decision_samples(decision, create_input_vector):
    (
        full_team_picks,
        team_picks,
        full_opponent_picks,
        opponent_picks,
        picked_hero,
        win,
    ) = decision

    augmented_samples = []

    own_picks_list = [pick for pick in full_team_picks]
    enemy_picks_list = [pick for pick in full_opponent_picks]

    total_own = len(own_picks_list)
    total_enemy = len(enemy_picks_list)

    # Your team's perspective: Cumulative with phased opponent inclusion
    for i in range(total_own):
        # Cumulative own picks up to but not including current (prior knowledge)
        prior_own_picks = own_picks_list[:i]

        # Phased opponent picks: Simulate Dota phases (empty early, then add)
        if i == 0:
            prior_enemy_picks = []  # No opponents visible yet
        elif i <= 2:
            prior_enemy_picks = enemy_picks_list[
                : min(2, total_enemy)
            ]  # Early phase: up to 2
        else:
            prior_enemy_picks = enemy_picks_list[
                : min(4, total_enemy)
            ]  # Later: up to 4

        # Actual pick is the current one
        actual_pick = own_picks_list[i]

        vec = create_input_vector(
            prior_own_picks, prior_enemy_picks, actual_pick
        )
        augmented_samples.append((vec, win))

    # Opponent's perspective: Swap and invert win
    inverted_win = 1 - win
    own_picks_list_inv = enemy_picks_list
    enemy_picks_list_inv = own_picks_list
    total_own_inv = len(own_picks_list_inv)
    total_enemy_inv = len(enemy_picks_list_inv)

    for i in range(total_own_inv):
        prior_own_picks = own_picks_list_inv[:i]

        if i == 0:
            prior_enemy_picks = []
        elif i <= 2:
            prior_enemy_picks = enemy_picks_list_inv[: min(2, total_enemy_inv)]
        else:
            prior_enemy_picks = enemy_picks_list_inv[: min(4, total_enemy_inv)]

        actual_pick = own_picks_list_inv[i]

        vec = create_input_vector(
            prior_own_picks, prior_enemy_picks, actual_pick
        )
        augmented_samples.append((vec, inverted_win))

    return augmented_samples


def stratified_split(
    x,
    y,
    test_size=0.2,
):
    # Group indices by label for stratification
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(y):
        label_to_indices[label].append(idx)

    train_indices = []
    val_indices = []
    for indices in label_to_indices.values():
        random.shuffle(indices)
        split_point = int(len(indices) * (1 - test_size))
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])

    # Shuffle final indices for randomness
    random.shuffle(train_indices)
    random.shuffle(val_indices)

    return train_indices, val_indices


def prepare_training_data(decisions, test_size=0.2):
    x = []
    y = []

    for (
        full_team_picks,
        team_picks,
        full_opponent_picks,
        opponent_picks,
        picked_hero,
        win,
    ) in decisions:
        input_vec = create_input_vector(
            team_picks, opponent_picks, picked_hero
        )
        x.append(input_vec)
        y.append(win)

        augmented = augment_decision_samples(
            (
                full_team_picks,
                team_picks,
                full_opponent_picks,
                opponent_picks,
                picked_hero,
                win,
            ),
            create_input_vector,
        )
        for sample_vec, label in augmented:
            x.append(sample_vec)
            y.append(label)

    logger.info(f"Augmented dataset size before split: {len(x)}")

    train_index, val_index = stratified_split(x, y, test_size=test_size)
    x_train = [x[i] for i in train_index]
    x_val = [x[i] for i in val_index]
    y_train = [y[i] for i in train_index]
    y_val = [y[i] for i in val_index]

    return x_train, x_val, y_train, y_val
