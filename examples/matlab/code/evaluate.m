function [F, G] = evaluate(X)
F(:,1) = X(:,1);
g = 1 + 9*mean(X(:,2:end),2);
h = 1 - (F(:,1)./g).^0.5;
F(:,2) = g.*h;
end


function [F, G] = evaluate_with_constr(X)
F(:,1) = X(:,1);
g = 1 + 9*mean(X(:,2:end),2);
h = 1 - (F(:,1)./g).^0.5;
F(:,2) = g.*h;

G(:,1) = X(:,1) - 1;


end