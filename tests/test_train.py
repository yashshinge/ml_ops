print('Working')
# #
# # from uniplot import plot
# # num_epochs = 2
# # test_accuracy = {1: 0, 2: 100}
# #
# # x, y = list(test_accuracy.keys()), list(test_accuracy.values())
# # plot(xs=x, ys=y, x_min=1, x_gridlines=[-1], x_max=num_epochs + 1,
# #      y_gridlines=y, y_min=max(min(y) - 5, 0), y_max=min(max(y) + 5, 100),
# #      lines=True, title="Plot for test accuracy (y) v/s Epochs (x)")
#
# import terminalplot as tp
test_accuracy = {1: 0, 2: 100}

x, y = list(test_accuracy.keys()), list(test_accuracy.values())

# tp.plot(x,y)



# import termplotlib as tpl
#
# fig = tpl.figure()
# fig.plot(x, y, label="data", width=50, height=15)
# fig.show()