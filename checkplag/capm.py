import numpy as np


class CAPM:
    """
    MM : Mean Matrix
    CM : Covariance Matrix
    EM : Expected Mean
    RR : Risk free return
    n : Number of stocks

    w_risk_free : weightage for market point
    w_risky : min risk for given mean
    ret : given mean
    ret_risky ; return of risky assets
    risk : risk of portfolio
    """

    def __init__(self, mean_matrix, covariance_matrix, expected_mean, risk_free_return, **kwargs):
        # Stocks data
        self.MM = mean_matrix
        self.RR = risk_free_return
        self.CM = covariance_matrix
        self.EM = expected_mean
        self.n = len(self.MM)

        # Prepare matrices
        identity = [1] * self.n
        self.CI = np.linalg.inv(self.CM)  # C inverse
        self.Identity = np.array(identity)

        # Optimal point ( Capital Market point )
        self.w_risk_free = None
        self.w_risky = None
        self.ret = None
        self.ret_risky = None
        self.risk = None

        # Derived portfolio ( Market point )
        self.ret_der = None
        self.risk_der = None

        # CAPM line
        self.slope = None

        # Compare point
        self.com_point = kwargs.get('compare_point', 0.15)

        # Prepare lines
        self._prepare()

    def _prepare(self):
        # Ideal market portfolio

        # Market point
        _a = (self.MM - self.RR * self.Identity) @ self.CI
        w_der = _a / (_a @ self.Identity.T)
        self.ret_der = self.MM @ w_der.T
        self.risk_der = np.sqrt(w_der @ self.CM @ w_der.T)

        # Optimal point ( Capital Market point )
        self.ret = self.EM

        w_risky_sum = (self.ret - self.RR) / (self.ret_der - self.RR)
        self.w_risky = w_risky_sum * w_der
        self.w_risk_free = 1 - w_risky_sum
        self.ret_risky = self.w_risky @ self.MM.T
        self.risk = w_risky_sum * self.risk_der

        # Capital market line
        self.capm_risk = np.arange(0, self.risk_der * 1.5, 0.02)
        self.slope = (self.ret_der - self.RR) / self.risk_der
        self.capm_ret = (self.slope * self.capm_risk) + self.RR

        # Comparison line
        self.line_comp = [0, (self.slope * self.com_point + self.RR) * 1.5]

    def plot(self, ax):
        # x axis
        ax.axhline(color='#000000')

        # y axis
        ax.axvline(color='#000000')
        ax.set_xlabel('Risk')
        ax.set_ylabel('Expected Return ')
        # Capital Market line
        ax.plot(self.capm_risk, self.capm_ret, label='Capital Market line', color="red", marker="_")

        # Risk free point
        ax.plot(0, self.RR, label='Risk free point', marker="o", color="blue")

        # Market point
        ax.plot(self.risk_der, self.ret_der, label='Market point', marker="o", color="orange")

        # Capital Market point
        ax.plot(self.risk, self.ret, label="Capital Market point", marker='o')

        # Comparision of risk
        ax.plot([self.com_point, self.com_point], self.line_comp, color='violet', linestyle='--',
                label='Comparison line')

        # Add a title
        ax.set_title('Capital assets pricing model')

        # Add a grid
        ax.grid(alpha=.4, linestyle=':')

        # Add a Legend
        ax.legend(prop={"size": 7}, loc="upper left")
