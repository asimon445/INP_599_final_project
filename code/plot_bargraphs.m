

%delta = linspace(-.3,.3,5);
delta = linspace(0,.6,5);
width = 0.1;
colors = {[0.95 0.05 0.05] [0.05 0.05 0.95]};   % [0.05 0.05 0.95] [0.95 0.05 0.95] [0.05 0.95 0.95]};

datas = {'acc_tree_used','acc_tree_shuffled_used'};

f1 = figure; 

for c = 1:2
    
    type = eval(sprintf('%s',datas{1,c}));
    all_feats(1,c) = {type};
    clear type

    boxplot(all_feats{1,c}(:,1:10),'symbol', '','Color',colors{c},'boxstyle','filled','position',(1:10)+delta(c),'widths',width);
    hold on;
    bh{c} = plot(nan,'Color',colors{c});
    ylabel('Classification Accuracy','FontWeight','bold','FontSize',16)
    xlabel('Diagnosis','FontWeight','bold','FontSize',16)

    if c == 2
        set(gca,'XTickLabel',header_used,'TickLabelInterpreter','none','FontSize',14);
    end    
end

h=gca; 
h.XAxis.TickLength = [0 0];

lines = findobj(gcf, 'type', 'line', 'Tag', 'Median');
set(lines, 'Color', 'k','linewidth',1.5);

[~, hobj, ~, ~] = legend([bh{:}],{'Classification accuracy','Chance'},'location','southeastoutside','FontSize',12);
hl = findobj(hobj,'type','line');
set(hl,'LineWidth',6);

