/* *****************************************************************************
 *
 * This wrapper is based on code authored by Stig Petersen, March 2014
 *
 * Xitari
 *
 * Copyright 2014 Google Inc.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License version 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 * *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare and
 *   the Reinforcement Learning and Artificial Intelligence Laboratory
 * Released under the GNU General Public License; see License.txt for details.
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 */

#ifndef __VIDEOCHESS_HPP__
#define __VIDEOCHESS_HPP__

#include "games/RomSettings.hpp"

namespace ale
{

    /* RL wrapper for Video Chess */
    class VideoChessSettings : public RomSettings
    {
    public:
        VideoChessSettings();

        // reset
        void reset() override;

        // is end of game
        bool isTerminal() const override;

        // get the most recently observed reward
        reward_t getReward() const override;

        // the rom-name
        const char* rom() const override { return "video_chess"; }

        // The md5 checksum of the ROM that this game supports
        const char* md5() const override { return "f0b7db930ca0e548c41a97160b9f6275"; }

        // create a new instance of the rom
        RomSettings* clone() const override;

        // is an action part of the minimal set?
        bool isMinimal(const Action& a) const override;

        // process the latest information from ALE
        void step(const stella::System& system) override;

        // saves the state of the rom settings
        void saveState(stella::Serializer& ser) override;

        // loads the state of the rom settings
        void loadState(stella::Deserializer& ser) override;

        int lives() override { return isTerminal() ? 0 : m_lives; }

        // Returns the five available game modes.
        ModeVect getAvailableModes() override;

        // Set the game mode.
        // The given mode must be one returned by the previous function.
        void setMode(game_mode_t, stella::System& system,
            std::unique_ptr<StellaEnvironmentWrapper> environment) override;

    private:
        bool m_terminal;
        reward_t m_reward;
        int m_lives;
    };

}  // namespace ale

#endif  // __VIDEOCHESS_HPP__
