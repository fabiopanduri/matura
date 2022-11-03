#!/bin/bash
# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of matura.
# matura is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, version 3.
# matura is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with
# matura. If not, see <https://www.gnu.org/licenses/>.

for ((i = 0 ; i < 1000; i++)); do
    python main.py dql -g pong -e 200 -s --reward-system v3
    echo $i
    sleep 2
done
