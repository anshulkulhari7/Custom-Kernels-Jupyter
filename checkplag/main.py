

from datetime import datetime
import requests
import os
from datetime import date
from urllib import request


import numpy as np
import cvxpy as cp
import pandas as pd


import matplotlib.gridspec as gd
import matplotlib.pyplot as plt

from marko import MarkowitzBullet
from capm import CAPM



stock_code = [
    
    # 'M&M.NS',
    #  'MRF.NS',
    'HINDUNILVR.NS',
     'RELIANCE.NS',
     'SBIN.NS',
     'TATAMOTORS.NS',
     'VOLTAS.NS',
    # 'BAJFINANCE.NS',
    # 'KOTAKBANK.NS',
     'TATASTEEL.NS',
    #  'TCS.NS',
     'CIPLA.NS',
    #  'LUPIN.NS',
    'TATACONSUM.NS',
    'SUNPHARMA.NS',
    'CYIENT.NS'

]
# Fetch data




CSV_CACHE_FOLDER = 'stockdata/'



def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks



def initialize(context):
    '''
    Called once at the very beginning of a backtest (and live trading). 
    Use this method to set up any bookkeeping variables.
    
    The context object is passed to all the other methods in your algorithm.
    Parameters
    context: An initialized and empty Python dictionary that has been 
             augmented so that properties can be accessed using dot 
             notation as well as the traditional bracket notation.
    
    Returns None
    '''
    # Register history container to keep a window of the last 100 prices.
    add_history(100, '1d', 'price')
    # Turn off the slippage model
    set_slippage(slippage.FixedSlippage(spread=0.0))
    # Set the commission model (Interactive Brokers Commission)
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0))
    context.tick = 0
    

def handle_data(context, data):
    '''
    Called when a market event occurs for any of the algorithm's 
    securities. 
    Parameters
    data: A dictionary keyed by security id containing the current 
          state of the securities in the algo's universe.
    context: The same context object from the initialize function.
             Stores the up to date portfolio as well as any state 
             variables defined.
    Returns None
    '''
    # Allow history to accumulate 100 days of prices before trading
    # and rebalance every day thereafter.
    context.tick += 1
    if context.tick < 100:
        return
    # Get rolling window of past prices and compute returns
    prices = history(100, '1d', 'price').dropna()
    returns = prices.pct_change().dropna()
    try:
        # Perform Markowitz-style portfolio optimization
        weights, _, _ = optimal_portfolio(returns.T)
        # Rebalance portfolio accordingly
        for stock, weight in zip(prices.columns, weights):
            order_target_percent(stock, weight)
    except ValueError as e:
        # Sometimes this error is thrown
        # ValueError: Rank(A) < p or Rank([P; A; G]) < n
        pass


class YahooFinanceData:
    def __init__(self, dataname, fromdate, todate, interval='1d'):
        posix = date(1970, 1, 1)

        # args for url
        if type(dataname) is list:
            self.dataname = dataname
        else:
            self.dataname = [dataname]
        self.period1 = int((fromdate.date() - posix).total_seconds())
        self.period2 = int((todate.date() - posix).total_seconds())
        self.interval = interval
        self.urlhist = 'https://finance.yahoo.com/quote/{}/history'
        self.urldown = 'https://query1.finance.yahoo.com/v7/finance/download/{}'

        # request session
        self.sess = requests.Session()
        self.sesskwargs = dict()
        self.retries = 4

        # finance data
        self.data = dict()

        # cache data folder
        if not os.path.isdir(CSV_CACHE_FOLDER):
            os.makedirs(CSV_CACHE_FOLDER)

    def prepare(self):
        required = set(self.dataname)

        for dname in self.dataname:
            # data and file
            stock_data = None
            file_name = '{}_{}_{}_{}.csv'.format(dname, self.period1, self.period2, self.interval)
            file_path = os.path.join(CSV_CACHE_FOLDER, file_name)
            error = None

            # find in cache first
            if os.path.isfile(file_path):
                log('{} : Found in cache'.format(dname), 2)
                required.remove(dname)
                continue

            # history URL to get crumb (API token)
            url = self.urlhist.format(dname)
            print(url)
            crumb = None
            for i in range(self.retries + 1):  # at least once
                resp = self.sess.get(url, **self.sesskwargs)
                if resp.status_code != requests.codes.ok:
                    print("error")
                    continue

                txt = resp.text
                i = txt.find('CrumbStore')
                if i == -1:
                    continue
                i = txt.find('crumb', i)
                if i == -1:
                    continue
                istart = txt.find('"', i + len('crumb') + 1)
                if istart == -1:
                    continue
                istart += 1
                iend = txt.find('"', istart)
                if iend == -1:
                    continue

                crumb = txt[istart:iend]
                crumb = crumb.encode('ascii').decode('unicode-escape')
                break

            if crumb is None:
                error = 'Crumb not found'
                log('{} : {}'.format(dname, error), 2)
                continue

            crumb = request.quote(crumb)

            # Download URL
            urld = self.urldown.format(dname)
            urlargs = list()
            urlargs.append('period2={}'.format(self.period2))
            urlargs.append('period1={}'.format(self.period1))
            urlargs.append('interval={}'.format(self.interval))
            urlargs.append('events=history')
            urlargs.append('crumb={}'.format(crumb))
            urld = '{}?{}'.format(urld, '&'.join(urlargs))

            for i in range(self.retries + 1):  # at least once
                resp = self.sess.get(urld, **self.sesskwargs)
                if resp.status_code != requests.codes.ok:
                    continue
                # ctype = resp.headers['Content-Type']
                # if 'text/csv' not in ctype:
                #     error = 'Wrong content type: %s' % ctype
                #     continue  # HTML returned? wrong url?
                stock_data = resp.text
                break

            if error:
                log('{} : {}'.format(dname, error), 2)
                continue

            log('{} : Downloaded and stored in cache'.format(dname), 2)
            with open(file_path, 'w') as f:
                f.write(stock_data)

            required.remove(dname)

        if len(required):
            raise ValueError("Not all stocks are downloaded")

        self._update()

    def _update(self):
        for dname in self.dataname:
            file_name = '{}.csv'.format(dname)
            print(file_name)
            file_path = os.path.join(CSV_CACHE_FOLDER, file_name)
            fileis = pd.read_csv(file_path)
            print(fileis)
            print("vaibhab")
            self.data[dname] = pd.read_csv(file_path)['Close']
            

        self.data = pd.DataFrame(self.data)
#############################################################################################################################3




VERBOSITY = 1


def log(msg, verbosity=1):
    if verbosity <= VERBOSITY:
        print(msg)


def mean_and_cov_matrix(data):
    # Returns and covariance matrix
    M = []
    C = []

    for key, value in data.items():
        ret_matrix = []
        zero_value = value[0]
        for val in value:
            ret_matrix.append((val-zero_value)/zero_value)
        data[key] = pd.Series(ret_matrix)

    for key1, value1 in data.items():
        Return = np.mean(value1)
        M.append(Return)

        CL = []
        for key2, value2 in data.items():
            cov = value1.cov(value2)
            CL.append(cov)
        C.append(CL)

    M = np.array(M)
    C = np.array(C)

    return M, C


def solvePortfolio(cov_matrix, mean_matrix, expected_mean=None):
    n = len(mean_matrix)
    w = cp.Variable(n)
    risk = cp.quad_form(w, cov_matrix)
    conditions = [
        sum(w) == 1,
        mean_matrix @ w.T == expected_mean
    ]
    if expected_mean is None:
        conditions = [
            sum(w) == 1
        ]

    prob = cp.Problem(cp.Minimize(risk), conditions)
    prob.solve()
    return w.value

#############################################################################################################################3

class PyfolioEngine:
    def __init__(self, data, expected_mean, risk_free_return, **kwargs):
        # Data
        self.data = data
        # self.ret_matrix, self.cov_matrix = mean_and_cov_matrix(self.data)

        # This calculates the mean vector
        returns_daily = data.pct_change()
        self.ret_matrix = np.array(returns_daily.mean() * 250)

        # This calculates the covariance matrix
        cov_daily = returns_daily.cov()
        self.cov_matrix = np.array(cov_daily * 250)

        # Markowitz bullet
        marko = kwargs.get('marko', dict())
        self.marko = MarkowitzBullet(self.ret_matrix, self.cov_matrix, expected_mean, **marko)

        # capital assets pricing model
        cap = kwargs.get('cap', dict())
        self.capm = CAPM(self.ret_matrix, self.cov_matrix, expected_mean, risk_free_return, **cap)

    def plot(self, show_marko=True, show_capm=True):
        """
        fig1: Markowitz bullet Figure
        fig2: Capital market line
        """
        # Part 1
        if show_marko:
            
            fig1 = plt.figure()
            fig1.canvas.set_window_title('Markowitz bullet')

            # Grid
            gs1 = gd.GridSpec(1, 1, figure=fig1)

            # plot
            ax1 = fig1.add_subplot(gs1[:, :])
            self.marko.plot(ax1)
            

        if show_capm:
            # Part 2
            # Figure
            fig2 = plt.figure()
            fig2.canvas.set_window_title('Capital assets pricing model')

            # Grid
            gs2 = gd.GridSpec(1, 1, figure=fig2)

            # plot
            ax2 = fig2.add_subplot(gs2[:, :])
            self.marko.plot(ax2, gp=100, line_only=True)
            self.capm.plot(ax2)

        # Show plot
        plt.show()

    def pprint(self, show_marko=True, show_capm=True):
        if show_marko:
            print('-----------------------------------------------------------------------------------')
            print('Weights for desired portfolio ( w/o risk free ) | Markowitz Theory                 ')
            print('-----------------------------------------------------------------------------------')
            for i, j in enumerate(zip(self.data, self.marko.w), 1):
                print('{:2} : {:10s} --> {:.6f}'.format(i, j[0], j[1]))
            print('-----------------------------------------------------------------------------------')
            print('Observations: ')
            print('1. For given return {:.2f}% minimum risk is {:.2f}%'.format(self.marko.ret*100, self.marko.risk*100))
            print()

        if show_capm:
            print('-----------------------------------------------------------------------------------')
            print('Weights for desired portfolio ( with risk free ) | Capital Assets Pricing Model    ')
            print('-----------------------------------------------------------------------------------')
            print('{:2} : {:10s} --> {:.6f}'.format(1, 'Risk Free', self.capm.w_risk_free))
            for i, j in enumerate(zip(self.data, self.capm.w_risky), 2):
                print('{:2} : {:10s} --> {:.6f}'.format(i, j[0], j[1]))
            print('-----------------------------------------------------------------------------------')
            print('Observations: ')
            print('1. For given return {:.2f}% minimum risk is {:.2f}%'.format(self.capm.ret * 100, self.capm.risk * 100))
            print('2. In given portfolio for {:.2f}% return obtained by market while, '.format(self.capm.ret_risky))
            print('   {:.2f}% return obtained by risk free assets'.format(self.capm.ret - self.capm.ret_risky))
            print('3. μ = {:.3f} σ + {:.3f}'.format(self.capm.slope, self.capm.RR))
#############################################################################################################3










data = YahooFinanceData(stock_code, fromdate=datetime(2021, 1, 1), todate=datetime(2020, 4, 1))
# data.prepare()
data._update()

# Let engine handle rest of it !!
engine = PyfolioEngine(data.data, 0.3, 0.07, marko={'mu_max': 1.5, 'gp_point': 70}, cap={'compare_point': 0.15})
engine.plot(show_marko=True, show_capm=True)
engine.pprint(show_marko=True, show_capm=True)
