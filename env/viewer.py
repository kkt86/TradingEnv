from collections import namedtuple
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from pandas.plotting import register_matplotlib_converters

style.use('ggplot')
register_matplotlib_converters()

VOLUME_CHART_HEIGHT = 0.33

Position = namedtuple('Position', ['type', 'shares', 'entry_price'])


class TradingGraph:
    """Trading visualization using matplotlib made to render OpenAI gym environments"""

    def __init__(self, df):
        self.df = df
        self.df = self.df.sort_values('time')

        # Create a figure on screen and set the title
        self.fig = plt.figure()

        # Create top subplot for net worth axis
        self.net_worth_ax = plt.subplot2grid(
            (6, 1), (0, 0), rowspan=2, colspan=1)

        # Create bottom subplot for shared price/volume axis
        self.price_ax = plt.subplot2grid(
            (6, 1), (2, 0), rowspan=8, colspan=1, sharex=self.net_worth_ax)

        # Create a new axis for volume which shares its x-axis with price
        self.volume_ax = self.price_ax.twinx()

        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.11, bottom=0.24,
                            right=0.90, top=0.90, wspace=0.2, hspace=0)

        # Show the graph without blocking the rest of the program
        plt.show(block=False)

    def render(self, current_step: int, portfolio: List[float],
               positions: List[Position], window_size: int = 200) -> None:
        """
        This method renders the last 'window_size' positions, price and volume entries and portfolio values
        :param current_step: actual step in the environment
        :param portfolio: list of portfolio values
        :param positions: list of positions taken
        :param window_size: length of the rendered history
        """
        value = round(portfolio[-1], 2)
        initial_value = round(portfolio[0], 2)
        profit_percent = round((value - initial_value) / initial_value * 100, 2)

        self.fig.suptitle('Portfolio value: $' + str(value) + ' | Profit: ' + str(profit_percent) + '%')

        window_start = max(current_step - window_size, 0)
        step_range = slice(window_start, current_step + 1)
        dates = self.df['time'].values[step_range]

        self._render_price(dates, positions, step_range, current_step)
        self._render_volume(step_range, dates)
        self._render_portfolio(step_range, dates, current_step, portfolio)

        # self._render_trades(step_range, trades)

        date_labels = self.df['date'].values[step_range]

        self.price_ax.set_xticklabels(
            date_labels, rotation=45, horizontalalignment='right')

        # Hide duplicate net worth date labels
        plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)

        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

    def close(self) -> None:
        """Closes the current plot"""
        plt.close()

    def _render_price(self, dates: List[np.datetime64], positions: List[Position],
                      step_range: slice, current_step: int) -> None:
        """
        This method adds price to the main plot
        :param dates: list of dates, to be used as x axis
        :param positions: list of positions, from which colors are extracted
        :param step_range: range to be used as reference for the plot
        :param current_step: last step to be plotted
        """
        self.price_ax.clear()

        segments = self._find_contiguous_colors(self._positions_to_colors(positions[step_range]))
        start = 0
        values = self.df['close'].values[step_range]
        for seg in segments:
            end = start + len(seg)
            self.price_ax.plot(dates[start:end + 1], values[start:end + 1], lw=2, c=seg[0])
            start = end

        last_date = self.df['time'].values[current_step]
        last_close = self.df['close'].values[current_step]
        last_high = self.df['high'].values[current_step]

        # Print the current price to the price axis
        self.price_ax.annotate('{0:.2f}'.format(last_close), (last_date, last_close),
                               xytext=(last_date, last_high),
                               bbox=dict(boxstyle='round',
                                         fc='w', ec='k', lw=1),
                               color="black",
                               fontsize="small")

        # Shift price axis up to give volume chart space
        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0])
                               * VOLUME_CHART_HEIGHT, ylim[1])

    def _render_volume(self, step_range: slice, dates: List[np.datetime64]) -> None:
        """
        This method plots volume underneath the price values
        :param step_range: range of values to be plotted
        :param dates: dates
        """
        self.volume_ax.clear()

        volume = np.array(self.df['volume'].values[step_range])

        self.volume_ax.plot(dates, volume, color='blue')
        self.volume_ax.fill_between(dates, volume, color='blue', alpha=0.5)

        self.volume_ax.set_ylim(0, max(volume) / VOLUME_CHART_HEIGHT)
        self.volume_ax.yaxis.set_ticks([])

    def _render_portfolio(self, step_range: slice, dates: List[np.datetime64],
                          current_step: int, portfolio: List[float]) -> None:
        """
        This method renders the portfolio values
        :param step_range: range of values to be plotted
        :param dates: dates
        :param current_step: current step to be used as last value
        :param portfolio: list of portfolio values
        """
        # Clear the frame rendered last step
        self.net_worth_ax.clear()
        print(type(current_step))
        # Plot net worths
        self.net_worth_ax.plot(
            dates, portfolio[step_range], label='Net Worth', color="g")

        # Show legend, which uses the label we defined for the plot above
        self.net_worth_ax.legend()
        legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_date = self.df['time'].values[current_step]
        last_net_worth = portfolio[current_step]

        # Annotate the current net worth on the net worth graph
        self.net_worth_ax.annotate('{0:.2f}'.format(last_net_worth), (last_date, last_net_worth),
                                   xytext=(last_date, last_net_worth),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")

        # Add space above and below min/max net worth
        self.net_worth_ax.set_ylim(
            min(portfolio) / 1.25, max(portfolio) * 1.25)

    @staticmethod
    def _positions_to_colors(positions: List[Position]) -> List[str]:
        """
        This method creates list of colors for the provided positions.
        If position is None, black is returned, else green if position is Long
        and red if position is SHORT
        :param positions: list of positions
        :return: list of colors
        """
        colors = []
        for position in positions:
            if not position:
                colors.append('k')
            else:
                if position.type == 'LONG':
                    colors.append('g')
                else:
                    colors.append('r')
        return colors

    @staticmethod
    def _find_contiguous_colors(colors: List[str]) -> List[List[str]]:
        """
        This method splits a list of colors into list of lists, where each list contains only same
        type of color
        :param colors: list of colors
        :return: list of lists of colors
        """
        segs = []
        curr_seg = []
        prev_color = ''
        for c in colors:
            if c == prev_color or prev_color == '':
                curr_seg.append(c)
            else:
                segs.append(curr_seg)
                curr_seg = []
                curr_seg.append(c)
            prev_color = c
        segs.append(curr_seg)  # the final one
        return segs
